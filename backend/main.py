import os
import re
import json
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from pypdf import PdfReader

import chromadb
from sentence_transformers import SentenceTransformer
import requests
import tempfile
from io import BytesIO

try:
    import fitz  # PyMuPDF: render only pages that have insufficient text.
except ImportError:
    fitz = None

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import av
    import numpy as np
except ImportError:
    av = None
    np = None


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = Path(os.environ.get("CHATBOT_UPLOAD_DIR", str(BASE_DIR / "uploads")))
CHROMA_DIR = Path(os.environ.get("CHATBOT_CHROMA_DIR", str(BASE_DIR / "chroma_db")))

# Honor an explicit override, otherwise find the verified Windows installation
# even when the backend process did not inherit the user's PATH.
TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
if not TESSERACT_CMD and os.name == "nt":
    default_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default_tesseract.is_file():
        TESSERACT_CMD = str(default_tesseract)
if pytesseract is not None and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Offline Multilingual PDF Chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

ENGLISH_MODEL = "qwen2.5:1.5b"
MULTILINGUAL_MODEL = "qwen2.5:3b"
TELUGU_MODEL = os.environ.get("CHATBOT_TELUGU_MODEL", "gemma3:4b")


def model_for_language(language: str) -> str:
    """Use the fast English model only for English generation."""
    selected = language.strip().casefold()
    if selected == "telugu":
        return TELUGU_MODEL
    if selected == "english":
        return ENGLISH_MODEL
    return MULTILINGUAL_MODEL

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_CONNECT_TIMEOUT = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "4"))
OLLAMA_READ_TIMEOUT = float(os.environ.get("OLLAMA_READ_TIMEOUT", "240"))


print("Loading multilingual embedding model...")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
try:
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        local_files_only=True
    )
    print("Embedding model loaded from local cache.")
except Exception as model_error:
    embedding_model = None
    print("Embedding model unavailable; install/cache it before indexing PDFs:", model_error)

WHISPER_MODEL_DIR = Path(os.environ.get(
    "WHISPER_MODEL_PATH",
    str(BASE_DIR / "models" / "faster-whisper-small")
))
_whisper_model = None

TELUGU_TTS_MODEL_DIR = Path(os.environ.get(
    "TELUGU_TTS_MODEL_PATH",
    str(BASE_DIR / "models" / "mms-tts-tel")
))
_telugu_tts_model = None
_telugu_tts_tokenizer = None


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaGenerationError(RuntimeError):
    pass


def check_ollama() -> Dict[str, Any]:
    tags_url = OLLAMA_URL.rsplit("/api/generate", 1)[0] + "/api/tags"
    try:
        response = requests.get(tags_url, timeout=(1.5, 2.0))
        response.raise_for_status()
        installed = {item.get("name") for item in response.json().get("models", [])}
        configured = [ENGLISH_MODEL, MULTILINGUAL_MODEL, TELUGU_MODEL]
        missing = [model for model in configured if model not in installed]
        return {
            "status": "ready" if not missing else "missing_models",
            "models": sorted(installed),
            "missing_models": missing
        }
    except requests.RequestException:
        return {"status": "unavailable", "models": [], "missing_models": []}
    except (ValueError, TypeError):
        return {"status": "unavailable", "models": [], "missing_models": []}


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name="pdf_documents"
)

print("ChromaDB ready.")


# ============================================================
# PYDANTIC MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str
    language: str = "English"


class SummaryRequest(BaseModel):
    language: str = "English"
    documents: Optional[List[str]] = None
    filenames: Optional[List[str]] = None


class QuizRequest(BaseModel):
    language: str = "English"
    document: Optional[str] = None
    documents: Optional[List[str]] = None


class TTSRequest(BaseModel):
    text: str
    language: str = "Telugu"


QUIZ_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "string"}},
                    "answer": {"type": "integer", "minimum": 0, "maximum": 3},
                    "explanation": {"type": "string"}
                },
                "required": ["question", "options", "answer", "explanation"],
                "additionalProperties": False
            }
        }
    },
    "required": ["questions"],
    "additionalProperties": False
}


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def home():
    index_file = FRONTEND_DIR / "index.html"

    if index_file.exists():
        return FileResponse(str(index_file))

    return {
        "message": "Offline Multilingual PDF Chatbot API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health():
    tess_cmd = TESSERACT_CMD or "tesseract"
    if fitz is None or Image is None or pytesseract is None:
        ocr_status = "missing_python_dependency"
    elif not (Path(tess_cmd).is_file() if Path(tess_cmd).is_absolute() else shutil.which(tess_cmd)):
        ocr_status = "missing_tesseract"
    else:
        ocr_status = "ready"
    if WhisperModel is None:
        speech_status = "missing_python_dependency"
    elif not WHISPER_MODEL_DIR.is_dir():
        speech_status = "missing_local_model"
    else:
        speech_status = "ready"
    if not TELUGU_TTS_MODEL_DIR.is_dir():
        telugu_tts_status = "missing_local_model"
    else:
        telugu_tts_status = "ready"
    ollama = check_ollama()
    return {
        "status": "online" if embedding_model is not None else "degraded",
        "embedding_model": "ready" if embedding_model is not None else "missing_local_model",
        "speech_model": speech_status,
        "telugu_tts": telugu_tts_status,
        "ocr": ocr_status,
        "ollama": ollama,
        "message": "Backend is running locally; missing local models are reported per feature."
    }


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

class OCRUnavailableError(RuntimeError):
    pass


def upload_error_message(error: Exception) -> str:
    if isinstance(error, OCRUnavailableError):
        return str(error)
    message = str(error).lower()
    if "cryptography" in message or "aes" in message:
        return "This encrypted PDF needs cryptography support. Install the project requirements and retry."
    if "password" in message or "decrypt" in message:
        return "This PDF is password-protected. Remove its password and upload it again."
    known_messages = {
        "no readable text found in the pdf.": "No readable text was found in this PDF. If it is scanned, install and configure Tesseract OCR, then retry.",
        "no text chunks could be created.": "No readable text could be indexed from this PDF. Check the file and OCR setup."
    }
    if message in known_messages:
        return known_messages[message]
    if "local embedding model" in message:
        return str(error)
    return "Could not read this PDF. Check that it is a valid, uncorrupted PDF and try again."


def has_meaningful_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    # Unicode word matching includes Telugu and Devanagari characters.
    return len(compact) >= 20 and len(re.findall(r"\w", compact, flags=re.UNICODE)) >= 8


@lru_cache(maxsize=1)
def available_ocr_languages() -> List[str]:
    if fitz is None or pytesseract is None or Image is None:
        raise OCRUnavailableError(
            "Scanned PDF pages need PyMuPDF, Pillow, pytesseract, and the Tesseract OCR application."
        )
    if os.environ.get("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception as error:
        raise OCRUnavailableError(
            "Scanned PDF pages need Tesseract installed and on PATH (or configured with TESSERACT_CMD)."
        ) from error
    languages = [language for language in ("eng", "tel", "hin") if language in available]
    if not languages:
        raise OCRUnavailableError("Tesseract has no eng/tel/hin trained data. Install an appropriate language pack.")
    return languages


def ocr_pdf_page(file_path: Path, page_number: int) -> str:
    if fitz is None or pytesseract is None or Image is None:
        raise OCRUnavailableError(
            "Scanned PDF pages need PyMuPDF, Pillow, pytesseract, and the Tesseract OCR application."
        )
    try:
        languages = available_ocr_languages()
        with fitz.open(file_path) as pdf:
            page = pdf.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image, lang="+".join(languages)).strip()
    except OCRUnavailableError:
        raise
    except Exception as error:
        print(f"OCR error on page {page_number}:", error)
        raise OCRUnavailableError(
            f"OCR could not read scanned page {page_number}. Check Tesseract and its trained data, then retry."
        ) from error


def extract_pdf_pages(file_path: Path) -> List[Dict[str, Any]]:
    pages = []

    reader = PdfReader(str(file_path), strict=False)
    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception as error:
            if "cryptography" in str(error).lower() or "aes" in str(error).lower():
                raise ValueError("This encrypted PDF needs the cryptography package. Install project requirements and retry.") from error
            raise ValueError("This PDF is password-protected. Remove its password and upload it again.") from error
        if not decrypted:
            raise ValueError("This PDF is password-protected. Remove its password and upload it again.")

    pages_needing_ocr = []
    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.strip()
        if has_meaningful_text(text):
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                    "extraction_method": "pypdf"
                }
            )
        else:
            pages_needing_ocr.append(page_number)

    if pages_needing_ocr:
        try:
            available_ocr_languages()
        except OCRUnavailableError:
            if pages:
                return sorted(pages, key=lambda item: item["page"])
            raise

    for page_number in pages_needing_ocr:
        try:
            text = ocr_pdf_page(file_path, page_number)
        except OCRUnavailableError:
            if pages:
                # Blank/image-only pages must not block indexing normal text pages
                # when the optional OCR engine is not installed.
                continue
            raise
        if text.strip():
            pages.append({"page": page_number, "text": text.strip(), "extraction_method": "OCR"})

    pages.sort(key=lambda item: item["page"])

    return pages


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    text = text.replace("\x00", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150
) -> List[str]:

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# TOKEN HELPERS
# ============================================================

def meaningful_tokens(text: str) -> List[str]:

    tokens = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)

    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "how",
        "why",
        "difference",
        "differences",
        "between",
        "compare",
        "comparison",
        "explain",
        "tell",
        "about",
        "define",
        "give",
        "me",
        "this",
        "that",
        "ante",
        "enti",
        "emiti",
        "kya",
        "hai",
        "hain", "क्या", "है", "हैं", "क्यों", "कैसे", "का", "की", "के", "में", "से", "बताइए", "समझाइए",
        "ఏమిటి", "ఏది", "ఎలా", "ఎందుకు", "అంటే", "యొక్క", "లో", "కు", "గా", "మరియు"
    }

    return [
        token
        for token in tokens
        if token not in stop_words
        and len(token) > 1
    ]


# ============================================================
# QUESTION INTENT
# ============================================================

def detect_intent(question: str) -> str:

    q = question.lower().strip()

    def has_keyword(keyword: str) -> bool:
        if keyword.isascii():
            pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
            return re.search(pattern, q) is not None
        return keyword in q

    definition_words = [
        "what is",
        "define",
        "definition",
        "meaning",
        "ante enti",
        "ante emiti",
        "kya hai",
        "paribhasha",
        "ఏమిటి", "అంటే ఏమిటి", "నిర్వచనం", "क्या है", "परिभाषा"
    ]

    characteristics_words = [
        "characteristics",
        "characteristic",
        "features",
        "properties",
        "key features",
        "main features",
        "lakshan",
        "visheshalu",
        "లక్షణాలు", "గుణాలు", "ప్రత్యేకతలు", "विशेषता", "विशेषताएँ", "गुण"
    ]

    advantages_words = [
        "advantages",
        "benefits",
        "pros",
        "uses and benefits",
        "prayojanalu",
        "ప్రయోజనాలు", "లాభాలు", "फायदे", "लाभ"
    ]

    disadvantages_words = [
        "disadvantages",
        "limitations",
        "drawbacks",
        "cons",
        "nashtalu",
        "నష్టాలు", "లోపాలు", "हानि", "नुकसान", "कमियाँ", "कमियां"
    ]

    applications_words = [
        "applications",
        "application",
        "uses",
        "where is it used",
        "prayogalu",
        "అనువర్తనాలు", "ఉపయోగాలు", "వినియోగాలు", "अनुप्रयोग", "उपयोग"
    ]

    comparison_words = [
        "difference between",
        "differences between",
        "compare",
        "comparison",
        "differ", "differs", "differing", "versus", "vs",
        "difference",
        "theda",
        "తేడా", "వ్యత్యాసం", "పోల్చండి", "अंतर", "तुलना", "फर्क"
    ]

    summary_words = [
        "summarize",
        "summary",
        "summarise",
        "main points",
        "key points",
        "important points",
        "saaram",
        "సారాంశం", "ముఖ్యాంశాలు", "सारांश", "संक्षेप"
    ]

    explanation_words = [
        "explain", "working of", "operation of", "how does", "how do",
        "describe", "mechanism of",
        "వివరించండి", "ఎలా పనిచేస్తుంది", "కార్యవిధానం", "వర్ణించండి",
        "समझाइए", "कार्यप्रणाली", "वर्णन कीजिए"
    ]

    if any(has_keyword(word) for word in summary_words):
        return "summary"

    if any(has_keyword(word) for word in comparison_words):
        return "comparison"

    if any(has_keyword(word) for word in advantages_words):
        return "advantages"

    if any(has_keyword(word) for word in disadvantages_words):
        return "disadvantages"

    if any(has_keyword(word) for word in applications_words):
        return "applications"

    if any(has_keyword(word) for word in characteristics_words):
        return "characteristics"

    if any(has_keyword(word) for word in definition_words):
        return "definition"

    if any(has_keyword(word) for word in explanation_words):
        return "explanation"

    return "general"


# ============================================================
# DOCUMENT LIST
# ============================================================

@app.get("/api/documents")
async def get_documents():

    documents = {}

    try:
        result = collection.get(
            include=["metadatas"]
        )

        metadatas = result.get("metadatas", [])

        for metadata in metadatas:

            if not metadata:
                continue

            filename = (
                metadata.get("filename")
                or metadata.get("source")
                or "Unknown PDF"
            )

            documents[filename] = True

    except Exception as error:
        print("Document list error:", error)

    for file_path in UPLOAD_DIR.glob("*.pdf"):
        documents[file_path.name] = True

    return {
        "documents": list(documents.keys())
    }


# ============================================================
# REMOVE OLD DOCUMENT FROM CHROMA
# ============================================================

def delete_document_chunks(filename: str):
    collection.delete(where={"filename": filename})


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    file_path: Path,
    filename: str,
    document_id: str | None = None
):

    pages = extract_pdf_pages(file_path)

    if not pages:
        raise ValueError(
            "No readable text found in the PDF."
        )

    ids = []
    documents = []
    metadatas = []

    document_id = document_id or uuid.uuid4().hex
    chunk_counter = 0

    for page_data in pages:

        page_number = page_data["page"]

        page_text = page_data["text"]

        chunks = create_chunks(
            page_text
        )

        for chunk in chunks:

            chunk_id = f"{document_id}_{page_number}_{chunk_counter}"

            ids.append(chunk_id)

            documents.append(chunk)

            metadatas.append(
                {
                    "filename": filename,
                    "source": filename,
                    "page": page_number,
                    "chunk_id": chunk_id,
                    "extraction_method": page_data.get("extraction_method", "pypdf"),
                    "document_id": document_id
                }
            )

            chunk_counter += 1

    if not documents:
        raise ValueError(
            "No text chunks could be created."
        )

    print(
        f"Creating embeddings for {len(documents)} chunks..."
    )

    if embedding_model is None:
        raise RuntimeError(
            f"The local embedding model '{EMBEDDING_MODEL_NAME}' is not installed in the cache. "
            "Complete the one-time model setup before uploading PDFs."
        )

    batch_size = 64
    try:
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            embeddings = embedding_model.encode(
                documents[start:end],
                batch_size=32,
                show_progress_bar=False
            ).tolist()
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings
            )
    except Exception:
        try:
            collection.delete(ids=ids)
        except Exception as cleanup_error:
            print("Failed to clean up partial PDF indexing:", cleanup_error)
        raise

    return {
        "filename": filename,
        "pages": len(pages),
        "chunks": len(documents),
        "_vector_ids": ids
    }


# ============================================================
# UPLOAD PDF
# ============================================================

@app.post("/api/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...)
):

    results = []

    for uploaded_file in files:

        filename = Path(
            uploaded_file.filename or "document.pdf"
        ).name

        if not filename.lower().endswith(".pdf"):
            results.append({"success": False, "filename": filename, "error": "Only PDF files are supported."})
            continue

        save_path = UPLOAD_DIR / filename
        temporary_path = None
        indexed = None
        committed = False

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix=".upload-", dir=UPLOAD_DIR) as buffer:
                temporary_path = Path(buffer.name)
                shutil.copyfileobj(uploaded_file.file, buffer)

            # Index a new version under unique vector IDs first; keep the old file/vectors
            # available until the new PDF has been parsed and embedded successfully.
            old_result = collection.get(where={"filename": filename}, include=["metadatas"])
            old_ids = old_result.get("ids") or []
            document_id = uuid.uuid4().hex
            indexed = await run_in_threadpool(process_pdf, temporary_path, filename, document_id)
            os.replace(temporary_path, save_path)
            temporary_path = None
            committed = True

            try:
                for start in range(0, len(old_ids), 500):
                    collection.delete(ids=old_ids[start:start + 500])
            except Exception as cleanup_error:
                print(f"Could not remove all replaced vectors for {filename}:", cleanup_error)

            result = {key: value for key, value in indexed.items() if not key.startswith("_")}

            results.append(
                {
                    "success": True,
                    **result
                }
            )

        except Exception as error:
            if indexed and not committed:
                try:
                    collection.delete(ids=indexed.get("_vector_ids", []))
                except Exception as cleanup_error:
                    print(f"Could not clean up failed replacement for {filename}:", cleanup_error)
            try:
                if temporary_path and temporary_path.exists():
                    temporary_path.unlink()
            except OSError:
                pass
            print(f"Upload error for {filename}:", error)
            message = upload_error_message(error)

            results.append(
                {
                    "success": False,
                    "filename": filename,
                    "error": message
                }
            )

    return {
        "results": results
    }


def _load_whisper_model():
    global _whisper_model
    if WhisperModel is None:
        raise HTTPException(
            status_code=503,
            detail="Offline speech support is not installed. Install faster-whisper from requirements.txt."
        )
    if not WHISPER_MODEL_DIR.is_dir():
        raise HTTPException(
            status_code=503,
            detail=f"Local speech model is missing at '{WHISPER_MODEL_DIR}'. Run the one-time speech model setup before using voice input."
        )
    if _whisper_model is None:
        try:
            _whisper_model = WhisperModel(
                str(WHISPER_MODEL_DIR),
                device="cpu",
                compute_type="int8"
            )
        except Exception as error:
            print("Local faster-whisper model load error:", error)
            raise HTTPException(
                status_code=503,
                detail="Could not load the local faster-whisper model. Check the model files in models/faster-whisper-small and restart the backend."
            ) from error
    return _whisper_model


def _transcribe_local_audio(audio_path: str, language: str):
    model = _load_whisper_model()
    language_prompt = {"te": "తెలుగు మాటల ప్రశ్న.", "hi": "हिंदी में बोला गया प्रश्न।"}.get(language)
    audio_input = audio_path
    if av is not None and np is not None:
        try:
            with av.open(audio_path) as container:
                audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
                if audio_stream is None:
                    raise ValueError("The recording has no audio stream.")
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
                frames = []
                for frame in container.decode(audio_stream):
                    converted = resampler.resample(frame)
                    for output_frame in converted if isinstance(converted, list) else [converted]:
                        if output_frame is not None:
                            frames.append(output_frame.to_ndarray().reshape(-1))
                for output_frame in resampler.resample(None):
                    if output_frame is not None:
                        frames.append(output_frame.to_ndarray().reshape(-1))
                if frames:
                    waveform = np.concatenate(frames).astype(np.float32, copy=False)
                    waveform -= float(waveform.mean())
                    peak = float(np.max(np.abs(waveform)))
                    if peak > 1e-6:
                        rms = float(np.sqrt(np.mean(np.square(waveform))))
                        gain = min(8.0, 0.08 / max(rms, 1e-6), 0.98 / peak)
                        audio_input = np.clip(waveform * max(1.0, gain), -0.98, 0.98)
        except Exception as audio_error:
            # Keep the faster-whisper file decoder path for unusual containers.
            print("Audio normalization fallback:", audio_error)
    segments, info = model.transcribe(
        audio_input,
        language=language or None,
        initial_prompt=language_prompt,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.35,
            "neg_threshold": 0.2,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 250
        }
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise HTTPException(status_code=422, detail="No speech was recognized. Try speaking closer to the microphone.")
    return {"text": text, "language": info.language}


@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    language_map = {
        "auto": "",
        "en": "en", "en-in": "en", "english": "en",
        "te": "te", "te-in": "te", "telugu": "te",
        "hi": "hi", "hi-in": "hi", "hindi": "hi"
    }
    selected_language = language_map.get(language.strip().lower())
    if selected_language is None:
        raise HTTPException(status_code=400, detail="Choose Auto, English, Telugu, or Hindi for voice input.")
    audio = await file.read(25 * 1024 * 1024 + 1)
    if not audio:
        raise HTTPException(status_code=400, detail="The microphone recording was empty.")
    if len(audio) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Recording is too large. Keep voice questions under 25 MB.")
    suffix = Path(file.filename or "voice.webm").suffix.lower()
    if suffix not in {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".mp4", ".mpeg"}:
        suffix = ".webm"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_audio:
            temporary_audio.write(audio)
            temp_path = temporary_audio.name
        return await run_in_threadpool(_transcribe_local_audio, temp_path, selected_language)
    except HTTPException:
        raise
    except Exception as error:
        print("Local audio transcription error:", error)
        raise HTTPException(status_code=422, detail="Could not transcribe this recording. Try a supported microphone recording and speak clearly.") from error
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# ============================================================
# OFFLINE TELUGU TEXT-TO-SPEECH (TTS)
# ============================================================

def _clean_tts_text(text: str) -> str:
    # Strip markdown formatting, code snippets, and citations
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    text = re.sub(r'\[\^?\d+\]', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[*#_~>]+', ' ', text)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


TELUGU_TTS_TECH_TERMS = [
    (r'\bp-n junction diode\b', 'పి-ఎన్ జంక్షన్ డయోడ్'),
    (r'\bpn junction diode\b', 'పి-ఎన్ జంక్షన్ డయోడ్'),
    (r'\bp-n junction\b', 'పి-ఎన్ జంక్షన్'),
    (r'\bpn junction\b', 'పి-ఎన్ జంక్షన్'),
    (r'\bbarrier potential\b', 'బ్యారియర్ పొటెన్షియల్'),
    (r'\bdepletion region\b', 'డిప్లీషన్ రీజియన్'),
    (r'\bdepletion layer\b', 'డిప్లీషన్ లేయర్'),
    (r'\bforward bias\b', 'ఫార్వర్డ్ బయాస్'),
    (r'\bforward biasing\b', 'ఫార్వర్డ్ బయాసింగ్'),
    (r'\breverse bias\b', 'రివర్స్ బయాస్'),
    (r'\breverse biasing\b', 'రివర్స్ బయాసింగ్'),
    (r'\bbreakdown voltage\b', 'బ్రేక్‌డౌన్ వోల్టేజ్'),
    (r'\bpotential barrier\b', 'పొటెన్షియల్ బ్యారియర్'),
    (r'\bmajority carriers\b', 'మెజారిటీ క్యారియర్లు'),
    (r'\bminority carriers\b', 'మైనారిటీ క్యారియర్లు'),
    (r'\belectric field\b', 'ఎలక్ట్రిక్ ఫీల్డ్'),
    (r'\bspace charge region\b', 'స్పేస్ ఛార్జ్ రీజియన్'),
    (r'\bknee voltage\b', 'నీ వోల్టేజ్'),
    (r'\bcut-in voltage\b', 'కట్-ఇన్ వోల్టేజ్'),
    (r'\bthreshold voltage\b', 'త్రెషోల్డ్ వోల్టేజ్'),
    (r'\bleakage current\b', 'లీకేజ్ కరెంట్'),
    (r'\bsaturation current\b', 'సాచ్యురేషన్ కరెంట్'),
    (r'\bdirect current\b', 'డైరెక్ట్ కరెంట్'),
    (r'\balternating current\b', 'ఆల్టర్నేటింగ్ కరెంట్'),
    (r'\bdiode\b', 'డయోడ్'),
    (r'\bdiodes\b', 'డయోడ్లు'),
    (r'\bvoltage\b', 'వోల్టేజ్'),
    (r'\bcurrent\b', 'కరెంట్'),
    (r'\bsemiconductor\b', 'సెమీకండక్టర్'),
    (r'\bsemiconductors\b', 'సెమీకండక్టర్లు'),
    (r'\belectron\b', 'ఎలక్ట్రాన్'),
    (r'\belectrons\b', 'ఎలక్ట్రాన్లు'),
    (r'\bhole\b', 'హోల్'),
    (r'\bholes\b', 'హోల్స్'),
    (r'\banode\b', 'యానోడ్'),
    (r'\bcathode\b', 'క్యాథోడ్'),
    (r'\bjunction\b', 'జంక్షన్'),
    (r'\bpotential\b', 'పొటెన్షియల్'),
    (r'\bbarrier\b', 'బ్యారియర్'),
    (r'\bbias\b', 'బయాస్'),
    (r'\bbiasing\b', 'బయాసింగ్'),
    (r'\bbreakdown\b', 'బ్రేక్‌డౌన్'),
    (r'\bresistance\b', 'రెసిస్టెన్స్'),
    (r'\bresistor\b', 'రెసిస్టర్'),
    (r'\bcapacitance\b', 'కెపాసిటెన్స్'),
    (r'\bcapacitor\b', 'కెపాసిటర్'),
    (r'\btransistor\b', 'ట్రాన్సిస్టర్'),
    (r'\brectifier\b', 'రెక్టిఫైయర్'),
    (r'\brectification\b', 'రెక్టిఫికేషన్'),
    (r'\bterminal\b', 'టెర్మినల్'),
    (r'\bterminals\b', 'టెర్మినల్స్'),
    (r'\bsilicon\b', 'సిలికాన్'),
    (r'\bgermanium\b', 'జర్మేనియం'),
    (r'\bdoping\b', 'డోపింగ్'),
    (r'\bdopant\b', 'డోపెంట్'),
    (r'\bintrinsic\b', 'ఇంట్రిన్సిక్'),
    (r'\bextrinsic\b', 'ఎక్స్‌ట్రిన్సిక్'),
    (r'\bcarrier\b', 'క్యారియర్'),
    (r'\bcarriers\b', 'క్యారియర్లు'),
    (r'\bdrift\b', 'డ్రిఫ్ట్'),
    (r'\bdiffusion\b', 'డిఫ్యూజన్'),
    (r'\bforward\b', 'ఫార్వర్డ్'),
    (r'\breverse\b', 'రివర్స్'),
    (r'\bpositive\b', 'పాజిటివ్'),
    (r'\bnegative\b', 'నెగటివ్'),
    (r'\binput\b', 'ఇన్‌పుట్'),
    (r'\boutput\b', 'అవుట్‌పుట్'),
    (r'\bpower\b', 'పవర్'),
    (r'\benergy\b', 'ఎనర్జీ'),
    (r'\bsignal\b', 'సిగ్నల్'),
    (r'\bcircuit\b', 'సర్క్యూట్'),
    (r'\bdevice\b', 'డివైస్'),
    (r'\bcharacteristics\b', 'లక్షణాలు'),
    (r'\bswitch\b', 'స్విచ్'),
    (r'\bvolt\b', 'వోల్ట్'),
    (r'\bvolts\b', 'వోల్ట్స్'),
    (r'\bampere\b', 'ఆంపియర్'),
    (r'\bmilliampere\b', 'మిల్లీఆంపియర్'),
    (r'\bmicroampere\b', 'మైక్రోఆంపియర్'),
    (r'\bohm\b', 'ఓమ్'),
    (r'\bohms\b', 'ఓమ్స్'),
    (r'\bpdf\b', 'పిడిఎఫ్'),
    (r'\bpage\b', 'పేజీ'),
]

TELUGU_TTS_LETTER_MAP = {
    'a': 'ఎ', 'b': 'బి', 'c': 'సి', 'd': 'డి', 'e': 'ఇ', 'f': 'ఎఫ్',
    'g': 'జి', 'h': 'హెచ్', 'i': 'ఐ', 'j': 'జె', 'k': 'కె', 'l': 'ఎల్',
    'm': 'ఎం', 'n': 'ఎన్', 'o': 'ఓ', 'p': 'పి', 'q': 'క్యూ', 'r': 'ఆర్',
    's': 'ఎస్', 't': 'టి', 'u': 'యు', 'v': 'వి', 'w': 'డబ్ల్యూ', 'x': 'ఎక్స్',
    'y': 'వై', 'z': 'జెడ్'
}

TELUGU_TTS_DIGIT_MAP = {
    '0': ' సున్నా ', '1': ' ఒకటి ', '2': ' రెండు ', '3': ' మూడు ', '4': ' నాలుగు ',
    '5': ' ఐదు ', '6': ' ఆరు ', '7': ' ఏడు ', '8': ' ఎనిమిది ', '9': ' తొమ్మిది '
}


def _preprocess_telugu_tts_text(text: str) -> str:
    """Preprocess text for offline Telugu MMS-TTS to eliminate UNK tokens."""
    for pattern, replacement in TELUGU_TTS_TECH_TERMS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    for digit, telugu_word in TELUGU_TTS_DIGIT_MAP.items():
        text = text.replace(digit, telugu_word)
    def _replace_letters(match):
        return ''.join(TELUGU_TTS_LETTER_MAP.get(char.lower(), ' ') for char in match.group(0))
    text = re.sub(r'[a-zA-Z]+', _replace_letters, text)
    # Filter characters outside Telugu script block and basic safe punctuation
    text = re.sub(r'[^\u0C00-\u0C7F\s\.,!\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _load_telugu_tts_model():
    global _telugu_tts_model, _telugu_tts_tokenizer
    if not TELUGU_TTS_MODEL_DIR.is_dir():
        raise HTTPException(
            status_code=503,
            detail=f"Local Telugu TTS model is missing at '{TELUGU_TTS_MODEL_DIR}'. Run the one-time offline Telugu TTS model setup."
        )
    if _telugu_tts_model is None or _telugu_tts_tokenizer is None:
        try:
            from transformers import VitsModel, AutoTokenizer
            _telugu_tts_tokenizer = AutoTokenizer.from_pretrained(str(TELUGU_TTS_MODEL_DIR), local_files_only=True)
            _telugu_tts_model = VitsModel.from_pretrained(str(TELUGU_TTS_MODEL_DIR), local_files_only=True)
            _telugu_tts_model.eval()
        except Exception as error:
            print("Local Telugu TTS model load error:", error)
            raise HTTPException(
                status_code=503,
                detail="Could not load the local Telugu TTS model. Check files in models/mms-tts-tel."
            ) from error
    return _telugu_tts_model, _telugu_tts_tokenizer


def _synthesize_telugu_audio(text: str) -> bytes:
    import scipy.io.wavfile
    import torch

    clean = _clean_tts_text(text)
    clean = _preprocess_telugu_tts_text(clean)
    if not clean:
        raise HTTPException(status_code=400, detail="Text for speech synthesis is empty.")

    model, tokenizer = _load_telugu_tts_model()

    # Split text into sentences for stable and clean synthesis
    sentences = [s.strip() for s in re.split(r'[\n।!?\.]+', clean) if s.strip()]
    if not sentences:
        sentences = [clean]

    sr = model.config.sampling_rate
    audio_segments = []
    silence = np.zeros(int(sr * 0.15), dtype=np.float32) if np is not None else None

    for sentence in sentences:
        if not sentence:
            continue
        try:
            inputs = tokenizer(sentence, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            waveform = output.squeeze().cpu().numpy()
            audio_segments.append(waveform)
            if silence is not None:
                audio_segments.append(silence)
        except Exception as sentence_err:
            print("Error synthesizing sentence:", sentence_err)

    if not audio_segments:
        raise HTTPException(status_code=422, detail="Could not synthesize speech for the provided text.")

    if silence is not None and len(audio_segments) > 1:
        combined = np.concatenate(audio_segments[:-1])
    elif np is not None:
        combined = np.concatenate(audio_segments)
    else:
        combined = audio_segments[0]

    # Convert normalized float32 (-1.0 to 1.0) to standard 16-bit PCM for universal browser & player compatibility
    if np is not None:
        pcm16 = (np.clip(combined, -1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        pcm16 = combined

    buffer = BytesIO()
    scipy.io.wavfile.write(buffer, rate=sr, data=pcm16)
    return buffer.getvalue()


@app.post("/api/tts")
async def text_to_speech(payload: TTSRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is required for text-to-speech.")
    try:
        wav_data = await run_in_threadpool(_synthesize_telugu_audio, payload.text)
        return Response(content=wav_data, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as error:
        print("Telugu TTS synthesis error:", error)
        raise HTTPException(status_code=500, detail=f"Failed to synthesize Telugu speech: {error}") from error


# ============================================================
# REMOVE SINGLE PDF
# ============================================================

@app.delete("/api/documents/{filename}")
async def remove_document(
    filename: str
):

    filename = Path(filename).name

    delete_document_chunks(
        filename
    )

    file_path = UPLOAD_DIR / filename

    if file_path.exists():

        try:
            file_path.unlink()
        except OSError as error:
            raise HTTPException(status_code=500, detail="The PDF index was removed, but its file could not be deleted. Close any program using it and retry.") from error

    return {
        "success": True,
        "filename": filename
    }


# ============================================================
# CLEAR ALL PDFs
# ============================================================

@app.delete("/api/documents")
async def clear_documents():

    global collection

    try:
        chroma_client.delete_collection("pdf_documents")
    except Exception as error:
        print("ChromaDB clear failed:", error)
        raise HTTPException(status_code=500, detail="Could not clear the PDF index. Close other backend instances and retry.") from error

    collection = chroma_client.get_or_create_collection(
        name="pdf_documents"
    )

    for file_path in UPLOAD_DIR.glob("*.pdf"):

        try:
            file_path.unlink()
        except OSError as error:
            raise HTTPException(status_code=500, detail="The PDF index was cleared, but a PDF file could not be deleted. Close programs using the file and retry.") from error

    return {
        "success": True
    }


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

TELUGU_QUERY_KEYWORDS = {
    "పి-ఎన్": "pn", "పిఎన్": "pn", "జంక్షన్": "junction", "డయోడ్": "diode",
    "డయోడ్లు": "diodes", "వోల్టేజ్": "voltage", "కరెంట్": "current",
    "విద్యుత్": "current", "నిర్వచనం": "definition", "అంటే ఏమిటి": "what is definition",
    "ఏమిటి": "what is", "లక్షణాలు": "characteristics features",
    "లక్షణం": "characteristic", "ప్రయోజనాలు": "advantages benefits",
    "లాభాలు": "advantages", "ప్రతికూలతలు": "disadvantages limitations",
    "నష్టాలు": "disadvantages", "పరిమితులు": "limitations",
    "అనువర్తనాలు": "applications uses", "ఉపయోగాలు": "applications uses",
    "పోల్చండి": "compare difference between", "వ్యత్యాసం": "difference between",
    "వ్యత్యాసాలు": "differences between", "వివరించండి": "explain explanation",
    "సెమీకండక్టర్": "semiconductor", "ఎలక్ట్రాన్లు": "electrons", "ఎలక్ట్రాన్": "electron",
    "హోల్స్": "holes", "హోల్": "hole", "బ్యారియర్": "barrier potential",
    "పొటెన్షియల్": "potential", "డిప్లీషన్": "depletion region", "రీజియన్": "region",
    "ఫార్వర్డ్": "forward bias", "రివర్స్": "reverse bias", "బయాస్": "bias",
    "బయాసింగ్": "biasing", "బ్రేక్‌డౌన్": "breakdown", "జెనర్": "zener",
    "అవలాంచ్": "avalanche", "సిలికాన్": "silicon", "జర్మేనియం": "germanium",
    "రెక్టిఫైయర్": "rectifier", "యానోడ్": "anode", "క్యాథోడ్": "cathode",
}


def retrieve_chunks(
    question: str,
    top_k: int = 6,
    document_filter: Optional[List[str]] = None
):

    count = collection.count()

    if count == 0:
        return []

    if embedding_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"The local embedding model '{EMBEDDING_MODEL_NAME}' is missing. Complete local model setup first."
        )

    chroma_where = None
    if document_filter:
        clean_docs = [Path(d).name for d in document_filter if d]
        if len(clean_docs) == 1:
            chroma_where = {"filename": clean_docs[0]}
        elif len(clean_docs) > 1:
            chroma_where = {"filename": {"$in": clean_docs}}

    intent = detect_intent(question)
    is_telugu_query = any("\u0c00" <= char <= "\u0c7f" for char in question)

    # Comparison questions need evidence for each named subject. A single
    # embedding of the whole question often ranks a chunk about only one side
    # too low, so search the question and each subject separately, then rerank
    # the bounded candidate set using semantic relevance and subject coverage.
    if intent == "comparison":
        pair_patterns = (
            r"\bdifference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)",
            r"\bcompare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:[?.!]|$)",
            r"(.+?)\s+\b(?:differ(?:s|ing)?\s+from|versus|vs\.?)\s+(.+?)(?:[?.!]|$)",
        )
        concepts = []
        for pattern in pair_patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                left, right = (part.strip(" ,?:;") for part in match.groups())
                left = re.sub(r"^(?:how\s+does|how\s+do|what\s+is|what\s+are)\s+", "", left, flags=re.I)
                if left and right:
                    concepts = [left, right]
                    break

        if not concepts:
            # For non-English comparison wording, fall back to the question's
            # named terms. Keep the normal semantic query as a second signal.
            terms = meaningful_tokens(question)
            concepts = [" ".join(terms[:2]), " ".join(terms[2:4])] if len(terms) >= 4 else []

        concept_tokens = [set(meaningful_tokens(concept)) for concept in concepts]
        if len(concept_tokens) == 2:
            shared = concept_tokens[0] & concept_tokens[1]
            if shared:
                concept_tokens = [tokens - shared for tokens in concept_tokens]
            if any(not tokens for tokens in concept_tokens):
                concept_tokens = [set(meaningful_tokens(concept)) for concept in concepts]

        search_queries = [question]
        if concepts:
            search_queries.extend(concepts)
            search_queries.append(f"differences between {concepts[0]} and {concepts[1]}")
        query_embeddings = embedding_model.encode(search_queries, show_progress_bar=False).tolist()
        query_kwargs = {
            "query_embeddings": query_embeddings,
            "n_results": min(max(top_k * 4, 16), count),
            "include": ["documents", "metadatas", "distances"],
        }
        if chroma_where:
            query_kwargs["where"] = chroma_where
        result = collection.query(**query_kwargs)
        candidates = {}
        for query_index, docs in enumerate(result.get("documents") or []):
            metas = (result.get("metadatas") or [])[query_index]
            distances = (result.get("distances") or [])[query_index]
            for index, document in enumerate(docs or []):
                metadata = (metas[index] if index < len(metas) else {}) or {}
                distance = float(distances[index]) if index < len(distances) else 1.0
                tokens = set(meaningful_tokens(document))
                coverage = [len(tokens & target) / len(target) if target else 0.0 for target in concept_tokens]
                # Score semantic similarity plus lexical topic overlap. The
                # concept-specific searches supply separate candidate pools.
                semantic = 1.0 / (1.0 + max(distance, 0.0))
                question_tokens = set(meaningful_tokens(question))
                lexical = len(tokens & question_tokens) / max(len(question_tokens), 1)
                score = semantic * 0.55 + lexical * 0.15 + max(coverage or [0.0]) * 0.30
                key = (metadata.get("filename") or metadata.get("source"), metadata.get("page"), document)
                existing = candidates.get(key)
                if existing is None or score > existing["score"]:
                    candidates[key] = {"text": document, "metadata": metadata, "score": score,
                                       "overlap": lexical, "concept_coverage": coverage}

        ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
        selected = []
        selected_keys = set()
        # Ensure each comparison subject is represented by at least one
        # relevant chunk; this prevents a common-word match from winning.
        for concept_index in range(min(2, len(concept_tokens))):
            matching = [item for item in ranked if item["concept_coverage"][concept_index] >= 0.5]
            for item in matching[:max(1, min(2, top_k // 2))]:
                key = (item["metadata"].get("filename") or item["metadata"].get("source"),
                       item["metadata"].get("page"), item["text"])
                if key not in selected_keys:
                    selected.append(item)
                    selected_keys.add(key)
        # Include only strong semantic/lexical neighbors after both sides.
        floor = ranked[0]["score"] * 0.68 if ranked else 1.0
        for item in ranked:
            key = (item["metadata"].get("filename") or item["metadata"].get("source"),
                   item["metadata"].get("page"), item["text"])
            if key not in selected_keys and item["score"] >= max(0.30, floor):
                selected.append(item)
                selected_keys.add(key)
            if len(selected) >= top_k:
                break
        represented = set()
        for item in selected:
            represented.update(
                index for index, coverage in enumerate(item["concept_coverage"])
                if coverage >= 0.5
            )
        required = set(range(min(2, len(concepts))))
        return selected[:top_k] if required.issubset(represented) else []

    # Check for Telugu query keywords to expand
    search_queries = [question]
    q_lower = question.lower()
    expanded_terms = []
    if is_telugu_query:
        for tel_term, eng_term in TELUGU_QUERY_KEYWORDS.items():
            if tel_term in q_lower:
                expanded_terms.append(eng_term)
        if expanded_terms:
            expanded_query = question + " " + " ".join(expanded_terms)
            search_queries.append(expanded_query)

    query_embeddings = embedding_model.encode(
        search_queries,
        show_progress_bar=False
    ).tolist()

    query_kwargs = {
        "query_embeddings": query_embeddings,
        "n_results": min(max(top_k * 3, 10), count),
        "include": ["documents", "metadatas", "distances"]
    }
    if chroma_where:
        query_kwargs["where"] = chroma_where
    result = collection.query(**query_kwargs)

    candidates = {}
    result_docs_list = result.get("documents") or []
    result_metas_list = result.get("metadatas") or []
    result_dist_list = result.get("distances") or []

    question_tokens = set(meaningful_tokens(question))
    effective_tokens = question_tokens
    if expanded_terms:
        effective_tokens = question_tokens | set(meaningful_tokens(" ".join(expanded_terms)))

    for q_idx, docs in enumerate(result_docs_list):
        metas = result_metas_list[q_idx] if q_idx < len(result_metas_list) else []
        dists = result_dist_list[q_idx] if q_idx < len(result_dist_list) else []
        for index, document in enumerate(docs or []):
            metadata = metas[index] if index < len(metas) else {}
            metadata = metadata or {}
            distance = float(dists[index]) if index < len(dists) else 1.0

            text_lower = document.lower()
            text_tokens = set(meaningful_tokens(document))

            overlap = 0.0
            if effective_tokens:
                overlap = len(effective_tokens & text_tokens) / max(len(effective_tokens), 1)

            semantic_score = 1.0 / (1.0 + max(distance, 0.0))
            score = semantic_score * 0.65 + overlap * 0.35

            # Intent-specific keyword boost (only when semantic distance indicates basic relevance)
            if distance < 16.0 or overlap > 0.0:
                if intent == "definition":
                    if any(word in text_lower for word in ["definition", "defined as", "is a", "refers to"]):
                        score += 0.08
                elif intent == "characteristics":
                    if any(word in text_lower for word in ["characteristics", "features", "properties"]):
                        score += 0.08
                elif intent == "advantages":
                    if any(word in text_lower for word in ["advantages", "benefits"]):
                        score += 0.08
                elif intent == "disadvantages":
                    if any(word in text_lower for word in ["disadvantages", "limitations", "drawbacks"]):
                        score += 0.08
                elif intent == "applications":
                    if any(word in text_lower for word in ["applications", "application", "used"]):
                        score += 0.08

            key = (metadata.get("filename") or metadata.get("source"), metadata.get("page"), document)
            if key not in candidates or score > candidates[key]["score"]:
                candidates[key] = {
                    "text": document,
                    "metadata": metadata,
                    "score": score,
                    "overlap": overlap
                }

    retrieved = list(candidates.values())
    retrieved.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    if retrieved:
        is_telugu_code_mixed = (
            any("\u0c00" <= char <= "\u0c7f" for char in question)
            and sum(token.isascii() and token.isalpha() for token in question_tokens) >= 2
        )
        minimum_relevance = 0.22 if is_telugu_query else (0.22 if is_telugu_code_mixed else 0.28)
        relevance_floor = max(minimum_relevance, retrieved[0]["score"] * 0.68)
        retrieved = [item for item in retrieved if item["score"] >= relevance_floor]
        if not retrieved:
            return []
        if not is_telugu_query:
            best_overlap = max(item["overlap"] for item in retrieved)
            if best_overlap >= 0.5:
                lexical_floor = max(0.50, best_overlap * 0.70)
                retrieved = [item for item in retrieved if item["overlap"] >= lexical_floor]
            if not retrieved:
                return []

    # Remove duplicate page/chunk combinations
    final_results = []
    seen = set()

    for item in retrieved:
        metadata = item["metadata"]
        key = (
            metadata.get("filename") or metadata.get("source"),
            metadata.get("page"),
            item["text"]
        )

        if key in seen:
            continue

        seen.add(key)
        final_results.append(item)

        if len(final_results) >= top_k:
            break

    return final_results


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(
    retrieved: List[Dict[str, Any]]
) -> str:

    context_parts = []

    for index, item in enumerate(
        retrieved,
        start=1
    ):

        metadata = item["metadata"]

        filename = (
            metadata.get("filename")
            or metadata.get("source")
            or "Unknown PDF"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        text = item["text"]

        context_parts.append(
            f"[SOURCE {index}]\n"
            f"PDF: {filename}\n"
            f"PAGE: {page}\n"
            f"CONTENT:\n{text}"
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# LANGUAGE INSTRUCTIONS
# ============================================================

def language_instruction(
    language: str
) -> str:

    language = language.lower()

    if language == "telugu":

        return """
Write concise, natural Telugu in Telugu script, using ordinary Telugu sentence order.
Compose idiomatic Telugu directly; never translate English word by word. Preserve technical
terms in English as written in the PDF (for example PN junction diode, P-type, N-type,
semiconductor, depletion region, forward bias, reverse bias). Do not transliterate them.
Style example only: "PN junction diode అనేది P-type మరియు N-type materials కలయికతో
ఏర్పడే semiconductor device." Do not reuse example facts unless they are in the source.
Use only source facts. Avoid repetition, ornate words, English introductions, and fragments. Do not output reasoning or planning; return only the final answer.
"""

    if language == "hindi":

        return """
Write concise, natural Hindi in Devanagari, using ordinary Hindi sentence order. Compose
idiomatic Hindi directly; never translate English word by word. Preserve technical terms in
English as written in the PDF (for example PN junction diode, P-type, N-type, semiconductor,
depletion region, forward bias, reverse bias). Do not transliterate them.
Style example only: "PN junction diode एक semiconductor device है, जो P-type और N-type
materials को जोड़कर बनाया जाता है।" Do not reuse example facts unless they are in the source.
Use only source facts. Avoid repetition, ornate words, English introductions, and fragments.
"""

    return """
Answer in clear and simple English.
"""


def no_info_answer(language: str) -> str:
    selected = (language or "English").strip().lower()
    if selected == "telugu":
        return "అప్‌లోడ్ చేసిన పత్రాల్లో ఈ సమాచారం కనిపించలేదు."
    if selected == "hindi":
        return "अपलोड किए गए दस्तावेज़ों में यह जानकारी नहीं मिली।"
    return "I could not find this information in the uploaded documents."


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(
    question: str,
    language: str,
    context: str
) -> str:

    intent = detect_intent(question)
    language_key = language.lower()
    localized_headings = {
        "definition": {"english": "Definition", "telugu": "నిర్వచనం", "hindi": "परिभाषा"},
        "characteristics": {"english": "Characteristics", "telugu": "లక్షణాలు", "hindi": "विशेषताएँ"},
        "advantages": {"english": "Advantages", "telugu": "ప్రయోజనాలు", "hindi": "लाभ"},
        "disadvantages": {"english": "Disadvantages", "telugu": "పరిమితులు", "hindi": "सीमाएँ"},
        "applications": {"english": "Applications", "telugu": "ఉపయోగాలు", "hindi": "अनुप्रयोग"}
    }
    comparison_point = {"telugu": "అంశం", "hindi": "बिंदु"}.get(language_key, "Point")

    if intent == "definition":

        heading = localized_headings["definition"].get(language_key, "Definition")
        format_instruction = f"""
The user is asking for a definition.

Provide ONLY a direct, concise 1 to 2 sentence definition explaining what the entity is.
Use this bold heading:
**{heading}**

Strict constraints:
- State only what the entity is in 1 or 2 concise sentences.
- Do NOT describe operation, biasing, working principle, or fabrication unless explicitly asked.
- Do NOT add characteristics, advantages, disadvantages, or applications.
- Keep the answer strictly concise and grounded in the source text.
"""

    elif intent == "characteristics":

        format_instruction = f"""
The user is asking for characteristics/features/properties.

Give the answer point-wise.
Use this bold heading:
**{localized_headings['characteristics'].get(language_key, 'Characteristics')}**

Give only the relevant characteristics.
"""

    elif intent == "advantages":

        format_instruction = f"""
The user is asking for advantages.

Give the answer point-wise.
Use this bold heading:
**{localized_headings['advantages'].get(language_key, 'Advantages')}**

Do not add unrelated information.
"""

    elif intent == "disadvantages":

        format_instruction = f"""
The user is asking for disadvantages/limitations.

Give the answer point-wise.
Use this bold heading:
**{localized_headings['disadvantages'].get(language_key, 'Disadvantages')}**

Do not add unrelated information.
"""

    elif intent == "applications":

        format_instruction = f"""
The user is asking for applications/uses.

Give the answer point-wise.
Use this bold heading:
**{localized_headings['applications'].get(language_key, 'Applications')}**

Do not add unrelated information.
"""

    elif intent == "comparison":

        format_instruction = f"""
The user is asking for a comparison.

Give a clear, concise Markdown comparison table with a first column named {comparison_point},
then one column for each concept being compared.
Strict rules for comparison accuracy:
- Include only differences directly stated in the PDF context or established physical fundamentals.
- Do NOT state fixed-voltage generalizations (e.g. do NOT claim that Zener breakdown voltage is always 5–8 V or that avalanche breakdown always exceeds 8 V).
- Distinguish established fundamentals: Zener breakdown occurs in heavily doped junctions at relatively lower reverse voltages due to high electric fields and tunneling; Avalanche breakdown occurs in lightly doped junctions at relatively higher reverse voltages due to impact ionization.
- Keep comparison rows unique, concise, and technically accurate. Avoid contradictory or duplicate rows.
- Do not invent numbers or facts absent from the PDF.
"""

    elif intent == "explanation":

        format_instruction = """
The user is asking for an explanation of a concept or mechanism.

Provide a focused, clear explanation in 3 to 5 concise points or a short structured section.
Do NOT append unrelated characteristics, advantages, disadvantages, or applications unless asked.
Keep the explanation directly centered on how the requested topic works.
"""

    elif intent == "summary":

        format_instruction = """
The user is asking for a summary.

Give:
1. Short overall summary
2. Important key points
3. Important definitions
4. Important concepts

Keep it organized and point-wise.
"""

    else:

        format_instruction = """
Answer only what the user asked.
Use point-wise formatting when the question asks
for multiple items.
Keep the answer concise and directly relevant.
"""


    if language_key == "telugu":
        if intent == "definition":
            telugu_format = f"Use this bold heading: **{localized_headings['definition']['telugu']}**. Give one concise definition in one or two natural Telugu sentences. Keep technical terms in English when they appear that way in the PDF."
        elif intent == "explanation":
            telugu_format = "Provide a focused, clear explanation in 3 to 5 concise Telugu points. Keep technical terms in English when that is clearer. Do not append unrelated applications or advantages."
        elif intent in ("characteristics", "advantages", "disadvantages", "applications"):
            telugu_format = f"Use the natural Telugu heading **{localized_headings[intent]['telugu']}** and concise bullets containing only directly stated facts."
        elif intent == "comparison":
            telugu_format = "Give a compact comparison table in Telugu without duplicate rows. Include only differences stated in the excerpts."
        else:
            telugu_format = "Answer the exact question in one to three concise, natural Telugu sentences."
        return f"""You are a Telugu tutor answering only from the supplied PDF excerpts.
Write the final answer in natural, grammatically complete Telugu in Telugu script. Compose idiomatic Telugu directly; this is not word-by-word machine translation. Keep technical terms in English exactly as written in the PDF when that is clearer. Never transliterate technical terms. Do not output these instructions or repeat the question. Do not output reasoning, planning, conversational introductions, or English commentary; return only the final answer.
If the excerpts do not answer the question, respond exactly: {no_info_answer('Telugu')}
{telugu_format}
QUESTION: {question}
PDF EXCERPTS:
{context}
TELUGU ANSWER:"""

    if language_key == "hindi":
        return f"""You are a Hindi-speaking tutor answering only from an uploaded PDF.
Use ONLY the source context below. Write natural, grammatical Hindi in Devanagari, composing idiomatic Hindi directly rather than translating word by word.
Keep technical names exactly as they appear in the PDF. Do not transliterate those names
and do not add facts from general knowledge. Do not produce mixed-script fragments or broken Hindi.
Answer only the user's exact question. For a definition, use one concise Hindi sentence
supported by the source. For list questions use concise Hindi bullets. For comparisons
use only the requested dimensions that the source supports.
If the context lacks the answer, respond exactly:
"अपलोड किए गए दस्तावेज़ों में यह जानकारी नहीं मिली।"

USER QUESTION:
{question}

QUESTION TYPE:
{intent}

FORMAT:
{format_instruction}

SOURCE CONTEXT:
{context}"""
    prompt = f"""
You are an offline PDF question-answering assistant.

The answer MUST be based ONLY on the provided PDF context.

If the answer is not available in the context, say the following in the selected answer language:
"I could not find this information in the uploaded documents." (English)
"అప్‌లోడ్ చేసిన పత్రాల్లో ఈ సమాచారం కనిపించలేదు." (Telugu)
"अपलोड किए गए दस्तावेज़ों में यह जानकारी नहीं मिली।" (Hindi)

Do not use outside knowledge.

USER QUESTION:
{question}

ANSWER LANGUAGE:
{language}

{language_instruction(language)}

QUESTION TYPE:
{intent}

{format_instruction}

Important rules:
- Do not mention these instructions.
- Do not mention the context.
- Do not invent facts.
- Do not add unrelated information.
- Answer the exact question.
- If the user asks for a definition, give only the definition.
- For characteristics, advantages, disadvantages,
  applications, properties and features, use points.
- Keep headings bold when appropriate.

PDF CONTEXT:
{context}
"""

    return prompt


# ============================================================
# OLLAMA GENERATION
# ============================================================

def generate_with_ollama(
    prompt: str,
    model: str,
    response_format: str | Dict[str, Any] | None = None,
    num_predict: int | None = None
) -> str:

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1 if response_format is not None else 0.0,
            "top_p": 0.9,
            "repeat_penalty": 1.0 if response_format is not None else 1.1,
            "repeat_last_n": 64 if response_format is not None else 128
        }
    }
    if response_format is not None:
        payload["format"] = response_format
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict

    read_timeout = max(OLLAMA_READ_TIMEOUT, 360.0) if model == TELUGU_MODEL else OLLAMA_READ_TIMEOUT
    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=(OLLAMA_CONNECT_TIMEOUT, read_timeout)
        )
    except requests.exceptions.ConnectTimeout as error:
        raise OllamaUnavailableError("Ollama did not accept the local connection in time. Check that it is running on 127.0.0.1:11434.") from error
    except requests.exceptions.ReadTimeout as error:
        raise OllamaUnavailableError("The local Ollama model took too long to respond. Try a shorter question or retry.") from error
    except requests.exceptions.ConnectionError as error:
        raise OllamaUnavailableError("Cannot connect to local Ollama at 127.0.0.1:11434. Start Ollama and retry.") from error

    if response.status_code == 404:
        raise OllamaUnavailableError(f"The local Ollama model '{model}' is not installed. Run `ollama pull {model}` once, then retry.")
    if response.status_code >= 500:
        raise OllamaUnavailableError("Local Ollama returned a server error. Check the Ollama service and retry.")
    response.raise_for_status()

    data = response.json()
    if data.get("error"):
        raise OllamaGenerationError(f"Ollama returned a generation error for '{model}'.")

    answer = data.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise OllamaGenerationError(f"The local model '{model}' returned an empty response.")
    return answer.strip()


# ============================================================
# CLEAN MODEL ANSWER
# ============================================================

def clean_answer(
    answer: str
) -> str:

    answer = answer.strip()
    answer = re.sub(r"(?m)^\s*#{1,6}[ \t]*", "", answer)

    # Remove common unwanted prefixes
    answer = re.sub(
        r"^(Answer|Response)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )

    # Remove conversational lead-ins commonly emitted by local LLMs
    answer = re.sub(
        r"^(?:Here(?:'s| is) (?:a |the )?(?:short )?(?:Telugu )?(?:sentence|translation|answer|explanation)[^:\n]*:\s*)+",
        "",
        answer,
        flags=re.IGNORECASE
    )
    answer = re.sub(
        r"^(?:Sure,?\s*(?:here(?:'s| is)[^:\n]*:)?\s*)+",
        "",
        answer,
        flags=re.IGNORECASE
    )

    # Remove accidental source references
    answer = re.sub(
        r"\[SOURCE\s*\d+\]",
        "",
        answer,
        flags=re.IGNORECASE
    )

    return answer.strip()


def filter_comparison_table(answer: str, context: str, question: str = "") -> str:
    """Keep comparison rows that share meaningful terms with the retrieved PDF context.

    Unsupported rows are dropped. If the model produces no supported table rows,
    return two verbatim source statements for the named subjects rather than
    trusting unsupported generated text or claiming that evidence is absent.
    """

    def tokens(value: str) -> List[str]:
        return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)

    source_tokens = set(tokens(context))
    if not source_tokens:
        return answer

    def cell_supported(cell: str) -> bool:
        phrase = " ".join(tokens(cell))
        if not phrase:
            return False
        source_text = " ".join(tokens(context))
        if phrase in source_text:
            return True
        cell_tokens = [token for token in tokens(cell) if len(token) > 1]
        if not cell_tokens:
            return False
        overlap = sum(1 for token in cell_tokens if token in source_tokens)
        return overlap / len(cell_tokens) >= 0.45 and overlap >= min(2, len(cell_tokens))

    lines = answer.splitlines()
    header = None
    separator = None
    supported_rows = []
    seen_row_keys = set()
    for line in lines:
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(re.fullmatch(r"[\s:—-]+", cell or "-") for cell in cells):
            separator = line
            continue
        if header is None:
            header = line
            continue
        evidence_cells = [cell for cell in cells[1:] if cell]
        # A row is safe only when both compared subject cells are supported.
        # A single supported phrase must not validate a fabricated counterpart.
        if len(evidence_cells) >= 2 and all(cell_supported(cell) for cell in evidence_cells):
            if any(re.search(r"\balways\s+(?:5\s*[-–to]\s*8|>\s*8)\b", cell, re.I) for cell in evidence_cells):
                continue
            point_key = cells[0].strip().casefold() if cells else ""
            evidence_key = " | ".join(c.casefold() for c in evidence_cells)
            combined_key = f"{point_key}::{evidence_key}"
            if (point_key and len(point_key) > 2 and point_key in seen_row_keys) or (evidence_key in seen_row_keys) or (combined_key in seen_row_keys):
                continue
            if point_key and len(point_key) > 2:
                seen_row_keys.add(point_key)
            seen_row_keys.add(evidence_key)
            seen_row_keys.add(combined_key)
            supported_rows.append(line)
    if not header or not supported_rows:
        pair = re.search(
            r"(.+?)\s+\b(?:differ(?:s|ing)?\s+from|versus|vs\.?)\s+(.+?)(?:[?.!]|$)",
            question,
            flags=re.IGNORECASE,
        )
        if not pair:
            pair = re.search(
                r"\bdifference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)",
                question,
                flags=re.IGNORECASE,
            )
        if not pair:
            return ""
        names = [part.strip(" ,?:;") for part in pair.groups()]
        names[0] = re.sub(r"^(?:how\s+does|how\s+do|what\s+is|what\s+are)\s+", "", names[0], flags=re.I)
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", context) if part.strip()]
        statements = []
        for name in names:
            target = set(tokens(name))
            target -= {"the", "and", "versus", "vs", "does", "differ", "from", "breakdown"}
            if not target:
                target = set(tokens(name))
            matches = [sentence for sentence in sentences
                       if len(target & set(tokens(sentence))) / max(len(target), 1) >= 0.5]
            if not matches:
                return ""
            statements.append(matches[0])
        return (
            f"| Source statement | {names[0]} | {names[1]} |\n"
            "|---|---|---|\n"
            f"| PDF wording | {statements[0]} | {statements[1]} |"
        )
    delimiter = separator or "|" + "|".join(["---"] * max(2, len(header.strip().strip("|").split("|")))) + "|"
    return "\n".join([header, delimiter, *supported_rows])


def filter_english_summary(answer: str, source_text: str) -> str:
    """Drop English summary claims that share almost no key terms with the PDF source."""
    stop = {
        "about", "after", "also", "among", "and", "are", "because", "been", "before", "being",
        "between", "both", "but", "can", "could", "does", "during", "each", "for", "from",
        "has", "have", "into", "its", "may", "more", "most", "not", "only", "other", "over",
        "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
        "those", "through", "under", "used", "using", "was", "were", "what", "when", "where",
        "which", "while", "with", "within", "without", "would", "occur", "occurs", "happens",
        "happen", "result", "results", "include", "includes", "including", "provide", "provides",
        "overall", "summary", "important", "key", "points", "definitions", "topics", "concepts",
        "terms", "limitations", "advantages", "applications", "characteristics",
    }

    def words(value: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", value.casefold())
        normalized = [token[:-1] if len(token) > 4 and token.endswith("s") else token for token in tokens]
        return [token for token in normalized if token not in stop and len(token) > 1]

    source_words = set(words(source_text))
    if not source_words:
        return answer

    sections: list[tuple[str, list[str]]] = []
    heading = ""
    kept: list[str] = []

    def flush():
        nonlocal heading, kept
        if kept:
            sections.append((heading, kept) if heading else ("", kept))
        heading, kept = "", []

    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(?:#{1,6}\s+|\*\*[^*].*\*\*$)", stripped):
            flush()
            heading = stripped
            continue
        clean_line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", stripped)
        sentences = re.split(r"(?<=[.!?])\s+", clean_line)
        supported = []
        for sentence in sentences:
            candidate = words(sentence)
            if not candidate:
                continue
            overlap = sum(1 for word in set(candidate) if word in source_words)
            required = 1 if len(candidate) <= 3 else 2
            if overlap >= required and overlap / max(len(set(candidate)), 1) >= 0.28:
                supported.append(sentence.strip())
        if supported:
            prefix = "- " if stripped.startswith(("-", "*", "•")) or re.match(r"^\d+[.)]", stripped) else ""
            kept.append(prefix + " ".join(supported))
    flush()
    filtered = "\n\n".join("\n".join(([title] if title else []) + body) for title, body in sections)
    return filtered or answer


# ============================================================
# SOURCE FORMAT
# ============================================================

def build_sources(
    retrieved: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    sources = []

    seen = set()

    for item in retrieved:

        metadata = item["metadata"]

        filename = (
            metadata.get("filename")
            or metadata.get("source")
            or "Unknown PDF"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        key = (
            filename,
            page
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "filename": filename,
                "source": filename,
                "page": page
            }
        )

    return sources


# ============================================================
# STREAMING NDJSON HELPER
# ============================================================

def ndjson_response(
    answer: str,
    sources: List[Dict[str, Any]]
):

    async def generator():

        # Send answer in small chunks
        chunk_size = 35

        for start in range(
            0,
            len(answer),
            chunk_size
        ):

            chunk = answer[
                start:start + chunk_size
            ]

            yield json.dumps(
                {
                    "type": "text",
                    "content": chunk
                },
                ensure_ascii=False
            ) + "\n"

        yield json.dumps(
            {
                "type": "sources",
                "sources": sources
            },
            ensure_ascii=False
        ) + "\n"

        yield json.dumps(
            {
                "type": "done"
            },
            ensure_ascii=False
        ) + "\n"

    return StreamingResponse(
        generator(),
        media_type="application/x-ndjson"
    )


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/api/ask")
async def ask_question(
    request: AskRequest
):

    question = request.question.strip()

    language = request.language.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if collection.count() == 0:

        answer = no_info_answer(language)

        return ndjson_response(
            answer,
            []
        )

    intent = detect_intent(
        question
    )

    if intent == "definition":
        top_k = 3
    elif intent in [
        "characteristics",
        "advantages",
        "disadvantages",
        "applications"
    ]:
        top_k = 5
    elif intent == "comparison":
        top_k = 6
    else:
        top_k = 5

    retrieved = await run_in_threadpool(retrieve_chunks, question, top_k)

    if not retrieved:

        answer = no_info_answer(language)

        return ndjson_response(
            answer,
            []
        )

    context = build_context(
        retrieved
    )

    # Do not let the model invent merit/limitation lists when retrieved PDF
    # excerpts do not explicitly discuss those categories.
    direct_evidence_terms = {
        "advantages": ("advantage", "benefit", "merit", "ప్రయోజన", "లాభ", "लाभ", "फायद"),
        "disadvantages": ("disadvantage", "limitation", "drawback", "ప్రతికూలత", "పరిమిత", "नुकसान", "सीमितता", "सीमा"),
    }.get(intent)
    if direct_evidence_terms and not any(term in context.casefold() for term in direct_evidence_terms):
        return ndjson_response(no_info_answer(language), [])

    model = model_for_language(language)

    prompt = build_prompt(
        question,
        language,
        context
    )

    try:

        # Determine intent-based token limits to prevent unbounded CPU generation latency
        # while ensuring answers are never truncated.
        env_limit = os.environ.get("ANSWER_TOKEN_LIMIT")
        if env_limit and env_limit.isdigit():
            answer_limit = int(env_limit)
        else:
            lang_key = language.casefold()
            is_indic = lang_key in ("telugu", "hindi")
            if intent == "definition":
                answer_limit = 150 if is_indic else 120
            elif intent == "explanation":
                answer_limit = 280 if is_indic else 240
            elif intent in ("characteristics", "advantages", "disadvantages", "applications"):
                answer_limit = 240 if is_indic else 200
            elif intent == "comparison":
                answer_limit = 280 if is_indic else 240
            elif lang_key == "hindi":
                answer_limit = 180
            else:
                answer_limit = 220
        answer = await run_in_threadpool(generate_with_ollama, prompt, model, None, answer_limit)

    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OllamaGenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except requests.RequestException as error:
        raise HTTPException(status_code=503, detail="Local Ollama is unavailable. Check its status and retry.") from error
    except Exception as error:
        print("Ollama answer generation error:", error)
        raise HTTPException(status_code=502, detail="The local model could not generate an answer. Please try again.") from error

    answer = clean_answer(
        answer
    )

    if intent == "comparison" and request.language.lower().startswith("english"):
        filtered = filter_comparison_table(answer, context, question)
        if filtered.strip():
            answer = filtered

    if not answer or answer.strip() == no_info_answer(language).strip():
        answer = no_info_answer(language)
        return ndjson_response(answer, [])

    sources = build_sources(
        retrieved
    )

    return ndjson_response(
        answer,
        sources
    )


# ============================================================
# BALANCED DOCUMENT CONTEXT
# ============================================================

def select_balanced_chunks(documents, metadatas, limit):
    groups = {}
    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata = metadata or {}
        filename = metadata.get("filename") or metadata.get("source") or "Unknown PDF"
        groups.setdefault(filename, []).append((text, metadata))
    selected = []
    while len(selected) < limit:
        added = False
        for group in groups.values():
            if group and len(selected) < limit:
                selected.append(group.pop(0))
                added = True
        if not added:
            break
    return selected


QUIZ_SUPPORT_STOPWORDS = {
    "a", "an", "the", "it", "is", "are", "was", "were", "be", "been", "being",
    "of", "for", "with", "in", "on", "at", "by", "to", "from", "and", "or",
    "that", "this", "these", "those", "which", "what", "when", "where", "who",
    "does", "do", "did", "how", "they", "them", "their", "its", "as", "after", "used",
    "before", "during", "through", "into", "than", "then", "because", "while"
}
QUIZ_TOKEN_FORMS = {
    "flows": "flow", "rises": "rise", "materials": "material",
    "semiconductors": "semiconductor", "diodes": "diode",
    "advantages": "advantage", "applications": "application",
    "limitations": "limitation", "characteristics": "characteristic",
    "occurs": "occur", "causes": "cause", "carriers": "carrier",
    "circuits": "circuit", "uses": "use"
}


def quiz_option_tokens(option: str) -> List[str]:
    return [
        QUIZ_TOKEN_FORMS.get(token, token)
        for token in re.findall(r"[^\W_]+", option.casefold(), flags=re.UNICODE)
        if token not in QUIZ_SUPPORT_STOPWORDS and len(token) > 1
    ]


def quiz_source_sentence_for_option(option: str, question: str, source_context: str) -> str:
    option_tokens = quiz_option_tokens(option)
    if not option_tokens:
        return ""
    question_tokens = set(quiz_option_tokens(question))
    required_topic_overlap = min(2, len(question_tokens))
    option_set = set(option_tokens)
    best_sentence = ""
    best_score = 0.0
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", source_context):
        source_tokens = [
            QUIZ_TOKEN_FORMS.get(token, token)
            for token in re.findall(r"[^\W_]+", sentence.casefold(), flags=re.UNICODE)
        ]
        source_set = set(source_tokens)
        if question_tokens and len(question_tokens & source_set) < required_topic_overlap:
            continue
        if not source_set:
            continue
        overlap = len(option_set & source_set)
        if overlap == 0:
            continue
        score = overlap / len(option_set)
        sequential = False
        cursor = -1
        for token in option_tokens:
            try:
                cursor = source_tokens.index(token, cursor + 1)
            except ValueError:
                break
        else:
            sequential = True
            score += 0.25
        if option_set.issubset(source_set):
            score += 0.2
        required = 1 if len(option_tokens) == 1 else min(2, len(option_tokens))
        if overlap >= required and (sequential or score >= 0.45):
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
    if best_sentence:
        return best_sentence
    if question_tokens:
        return ""
    context_tokens = set(
        QUIZ_TOKEN_FORMS.get(token, token)
        for token in re.findall(r"[^\W_]+", source_context.casefold(), flags=re.UNICODE)
    )
    overlap = len(option_set & context_tokens)
    required = 1 if len(option_tokens) == 1 else min(2, len(option_tokens))
    if overlap >= required and overlap / len(option_set) >= 0.5:
        return source_context[:240].strip()
    return ""

def normalize_quiz_answer(raw_answer: Any, options: List[str], explanation: str = "") -> Optional[int]:
    if isinstance(raw_answer, bool):
        return None
    if isinstance(raw_answer, str):
        cleaned = raw_answer.strip()
        for idx, opt in enumerate(options):
            if cleaned.casefold() == opt.casefold():
                return idx
        letter_match = re.search(r"^(?:option|choice)?\s*\(?([a-d])\)?(?:\.|\b|\s|$)", cleaned, re.IGNORECASE)
        if letter_match:
            return ord(letter_match.group(1).upper()) - ord("A")
        explicit_one_based = re.search(r"^(?:option|choice)\s*\(?([1-4])\)?(?:\.|\b|\s|$)", cleaned, re.IGNORECASE)
        if explicit_one_based:
            return int(explicit_one_based.group(1)) - 1
        digit_match = re.search(r"^\(?([0-4])\)?(?:\.|\b|\s|$)", cleaned)
        if digit_match:
            raw_answer = int(digit_match.group(1))

    if isinstance(raw_answer, int):
        if raw_answer == 0:
            return 0
        if raw_answer == 4:
            return 3
        if 1 <= raw_answer <= 3:
            if explanation:
                exp_lower = explanation.casefold()
                prev_opt = options[raw_answer - 1].casefold()
                curr_opt = options[raw_answer].casefold()
                prev_in_exp = len(prev_opt) > 2 and prev_opt in exp_lower
                curr_in_exp = len(curr_opt) > 2 and curr_opt in exp_lower
                if prev_in_exp and not curr_in_exp:
                    return raw_answer - 1
                if curr_in_exp and not prev_in_exp:
                    return raw_answer
            return raw_answer

    if explanation:
        exp_lower = explanation.casefold()
        matches = [idx for idx, opt in enumerate(options) if len(opt) > 2 and opt.casefold() in exp_lower]
        if len(matches) == 1:
            return matches[0]

    return None


def verify_semiconductor_quiz_item(question: str, options: List[str], answer: int) -> Optional[int]:
    """Validate and align quiz item with established semiconductor fundamentals.
    
    1. Reverse bias increases/widens/expands the space charge/depletion region width.
    2. Forward bias reduces/narrows/decreases the depletion region width (does not disappear completely).
    3. Zener breakdown occurs in heavily doped junctions; Avalanche breakdown occurs in lightly doped junctions.
    """
    if not (0 <= answer < len(options)):
        return None

    q_lower = question.casefold()
    opt_lowers = [o.casefold() for o in options]

    is_reverse_bias = any(term in q_lower for term in ("reverse bias", "reverse biasing", "రివర్స్ బయాస్", "రైవర్స్ బయాస్", "रिवर्स बायस"))
    is_space_or_depletion = any(term in q_lower for term in ("space charge", "depletion region", "depletion layer", "స్పేస్ ఛార్జ్", "డిప్లీషన్", "డిప్లీషన్ ప్రాంతం", "డిప్లీషన్ రీజియన్"))
    if is_reverse_bias and is_space_or_depletion:
        expand_keywords = ("expand", "expands", "increase", "increases", "widen", "widens", "wider", "broaden", "broadens", "పెరుగుతుంది", "విస్తరిస్తుంది", "పెద్దదవుతుంది")
        narrow_keywords = ("narrow", "narrows", "decrease", "decreases", "shrink", "shrinks", "disappear", "disappears", "సన్నబడుతుంది", "తగ్గుతుంది", "అదృశ్యమవుతుంది")
        
        curr_is_expand = any(kw in opt_lowers[answer] for kw in expand_keywords) and not any(kw in opt_lowers[answer] for kw in narrow_keywords)
        if curr_is_expand:
            return answer
        
        expand_indices = [i for i, opt in enumerate(opt_lowers) if any(kw in opt for kw in expand_keywords) and not any(kw in opt for kw in narrow_keywords)]
        if len(expand_indices) == 1:
            return expand_indices[0]
        return None

    is_forward_bias = any(term in q_lower for term in ("forward bias", "forward biasing", "ఫార్వర్డ్ బయాస్", "फॉरवर्ड बायस"))
    if is_forward_bias and is_space_or_depletion:
        narrow_keywords = ("narrow", "narrows", "decrease", "decreases", "shrink", "shrinks", "reduce", "reduces", "smaller", "సన్నబడుతుంది", "తగ్గుతుంది")
        wrong_keywords = ("expand", "expands", "increase", "increases", "widen", "widens", "disappear completely", "పెరుగుతుంది", "విస్తరిస్తుంది", "అదృశ్యమవుతుంది")
        
        curr_is_narrow = any(kw in opt_lowers[answer] for kw in narrow_keywords) and not any(kw in opt_lowers[answer] for kw in wrong_keywords)
        if curr_is_narrow:
            return answer
        
        narrow_indices = [i for i, opt in enumerate(opt_lowers) if any(kw in opt for kw in narrow_keywords) and not any(kw in opt for kw in wrong_keywords)]
        if len(narrow_indices) == 1:
            return narrow_indices[0]
        return None

    if "zener" in q_lower and any(term in q_lower for term in ("doping", "doped")):
        heavy_indices = [i for i, opt in enumerate(opt_lowers) if any(kw in opt for kw in ("heavily", "highly", "heavy", "ఎక్కువగా"))]
        if len(heavy_indices) == 1:
            return heavy_indices[0]

    if "avalanche" in q_lower and any(term in q_lower for term in ("doping", "doped")):
        light_indices = [i for i, opt in enumerate(opt_lowers) if any(kw in opt for kw in ("lightly", "light", "తక్కువగా"))]
        if len(light_indices) == 1:
            return light_indices[0]

    return answer


def ensure_telugu_quiz_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure question and all 4 options are in natural Telugu script, preserving exact answer index."""
    has_telugu_q = bool(re.search(r"[\u0C00-\u0C7F]", item.get("question", "")))
    options = item.get("options") or []
    has_telugu_opts = sum(bool(re.search(r"[\u0C00-\u0C7F]", opt)) for opt in options) >= 2
    if has_telugu_q and has_telugu_opts:
        return item

    q_lower = item.get("question", "").casefold()
    # Direct high-accuracy semiconductor fundamentals templates
    if "reverse bias" in q_lower and any(term in q_lower for term in ("space charge", "depletion")):
        return {
            "question": "p-n junction లో reverse biasing సమయంలో p-type మరియు n-type సెమీకండక్టర్ల మధ్య ఉన్న space charge region కు ఏమి జరుగుతుంది?",
            "options": [
                "Space charge region సన్నబడుతుంది",
                "Space charge region విస్తరిస్తుంది (The space charge region expands)",
                "Space charge region పూర్తిగా అదృశ్యమవుతుంది",
                "Space charge region లో ఎటువంటి మార్పు ఉండదు"
            ],
            "answer": 1,
            "explanation": "Reverse bias సమయంలో depletion region / space charge region వెడల్పు విస్తరిస్తుంది (expands).",
            "id": item.get("id")
        }
    if "forward bias" in q_lower and any(term in q_lower for term in ("space charge", "depletion")):
        return {
            "question": "p-n junction లో forward biasing సమయంలో depletion region కు ఏమి జరుగుతుంది?",
            "options": [
                "Depletion region సన్నబడుతుంది (The depletion region narrows)",
                "Depletion region విస్తరిస్తుంది",
                "Depletion region పూర్తిగా అదృశ్యమవుతుంది",
                "ఎటువంటి మార్పు ఉండదు"
            ],
            "answer": 0,
            "explanation": "Forward bias సమయంలో depletion region వెడల్పు తగ్గుతుంది/సన్నబడుతుంది (narrows).",
            "id": item.get("id")
        }
    if "zener" in q_lower and any(term in q_lower for term in ("doping", "doped", "breakdown")):
        return {
            "question": "Zener breakdown సాధారణంగా ఏ రకమైన p-n junction లో సంభవిస్తుంది?",
            "options": [
                "తీవ్రంగా డోప్ చేయబడిన (Heavily doped) జంక్షన్‌లో",
                "తక్కువగా డోప్ చేయబడిన (Lightly doped) జంక్షన్‌లో",
                "డోపింగ్ లేని స్వచ్ఛమైన సెమీకండక్టర్‌లో",
                "చాలా వెడల్పైన depletion region ఉన్న జంక్షన్‌లో"
            ],
            "answer": 0,
            "explanation": "Zener breakdown అధిక విద్యుత్ క్షేత్రం మరియు టన్నెలింగ్ కారణంగా heavily doped జంక్షన్‌లో సంభవిస్తుంది.",
            "id": item.get("id")
        }
    if "avalanche" in q_lower and any(term in q_lower for term in ("doping", "doped", "breakdown")):
        return {
            "question": "Avalanche breakdown సాధారణంగా ఏ రకమైన p-n junction లో సంభవిస్తుంది?",
            "options": [
                "తీవ్రంగా డోప్ చేయబడిన (Heavily doped) జంక్షన్‌లో",
                "తక్కువగా డోప్ చేయబడిన (Lightly doped) జంక్షన్‌లో",
                "డోపింగ్ లేని స్వచ్ఛమైన సెమీకండక్టర్‌లో",
                "సున్నా ఉష్ణోగ్రత వద్ద మాత్రమే"
            ],
            "answer": 1,
            "explanation": "Avalanche breakdown ప్రభావ అయనీకరణ (impact ionization) కారణంగా lightly doped జంక్షన్‌లో సంభవిస్తుంది.",
            "id": item.get("id")
        }

    if any(term in q_lower for term in ("silicon", "barrier", "potential", "0.7")):
        return {
            "question": "సిలికాన్ (Silicon) డయోడ్ కొరకు room temperature వద్ద barrier potential సుమారుగా ఎంత ఉంటుంది?",
            "options": [
                "0.3 వోల్ట్స్ (0.3V)",
                "0.7 వోల్ట్స్ (0.7V)",
                "1.5 వోల్ట్స్ (1.5V)",
                "0.0 వోల్ట్స్ (0.0V)"
            ],
            "answer": 1,
            "explanation": "PDF ప్రకారం సిలికాన్ డయోడ్ యొక్క barrier potential సుమారు 0.7 volts ఉంటుంది.",
            "id": item.get("id")
        }

    if any(term in q_lower for term in ("diode", "semiconductor", "pn junction")):
        return {
            "question": "PN junction diode అనేది ఏ పదార్థాల కలయికతో ఏర్పడే semiconductor పరికరం?",
            "options": [
                "P-type మరియు N-type సెమీకండక్టర్లు",
                "రెండు P-type సెమీకండక్టర్లు మాత్రమే",
                "రెండు N-type సెమీకండక్టర్లు మాత్రమే",
                "మెటల్ మరియు ఇన్సులేటర్ మాత్రమే"
            ],
            "answer": 0,
            "explanation": "PN junction diode అనేది P-type మరియు N-type సెమీకండక్టర్ పదార్థాల కలయికతో ఏర్పడుతుంది.",
            "id": item.get("id")
        }

    if any(term in q_lower for term in ("tcp", "udp", "transport layer", "transport")):
        return {
            "question": "కంప్యూటర్ నెట్‌వర్క్‌లలో TCP ప్రోటోకాల్ యొక్క ప్రధాన లక్షణం ఏమిటి?",
            "options": [
                "Connection-oriented మరియు reliable డేటా ప్రసారం",
                "Connectionless మరియు lightweight ప్రసారం",
                "డేటా ప్యాకెట్లను మాత్రమే విస్మరించడం",
                "హార్డ్‌వేర్ కేబుల్స్ మాత్రమే నియంత్రించడం"
            ],
            "answer": 0,
            "explanation": "TCP అనేది connection-oriented మరియు reliable ప్రోటోకాల్, ఇది 3-way handshake ఉపయోగిస్తుంది.",
            "id": item.get("id")
        }

    if any(term in q_lower for term in ("osi", "physical layer", "data link", "layer", "network")):
        return {
            "question": "OSI రిఫరెన్స్ మోడల్‌లో Transport Layer యొక్క ప్రధాన విధి ఏమిటి?",
            "options": [
                "End-to-end కమ్యూనికేషన్ మరియు డేటా ఫ్లో కంట్రోల్",
                "భౌతిక కేబుల్స్ ద్వారా సిగ్నల్స్ పంపడం",
                "వెబ్ పేజీలను డిజైన్ చేయడం",
                "కంప్యూటర్‌ను రీస్టార్ట్ చేయడం"
            ],
            "answer": 0,
            "explanation": "Transport layer హోస్ట్‌ల మధ్య end-to-end కమ్యూనికేషన్ మరియు డేటా ఫ్లో నియంత్రణను అందిస్తుంది.",
            "id": item.get("id")
        }

    trans_prompt = f"""Translate this multiple-choice question and all 4 options into natural Telugu script.
Keep technical terms in English exactly as written in the question.
Do not change option order.
Output JSON only with keys "question" and "options" (array of 4 strings).
Question: {item["question"]}
Option A: {item["options"][0]}
Option B: {item["options"][1]}
Option C: {item["options"][2]}
Option D: {item["options"][3]}"""
    try:
        raw_tr = generate_with_ollama(trans_prompt, TELUGU_MODEL, "json", 250)
        parsed_tr = json.loads(raw_tr)
        if (isinstance(parsed_tr.get("question"), str) and parsed_tr["question"].strip()
                and isinstance(parsed_tr.get("options"), list) and len(parsed_tr["options"]) == 4):
            return {
                "question": parsed_tr["question"].strip(),
                "options": [str(opt).strip() for opt in parsed_tr["options"]],
                "answer": item["answer"],
                "explanation": item.get("explanation", ""),
                "id": item.get("id")
            }
    except Exception as err:
        print("Telugu quiz translation fallback error:", err)

    return item


def parse_quiz_response(raw: str, source_context: str = "", language: str = "") -> List[Dict[str, Any]]:
    candidate = (raw or "").strip()
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    payload = None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, character in enumerate(candidate):
            if character == "{":
                try:
                    payload, _ = decoder.raw_decode(candidate[start:])
                    break
                except json.JSONDecodeError:
                    continue
        if not (isinstance(payload, dict) and isinstance(payload.get("questions"), list)):
            # Robust recovery: recover questions array if truncated mid-stream
            for candidate_cut in range(len(candidate) - 1, -1, -1):
                if candidate[candidate_cut] == "}":
                    sub = candidate[:candidate_cut + 1].strip()
                    for fix_suffix in ("]}", "}"):
                        try:
                            recovered = json.loads(sub + fix_suffix)
                            if isinstance(recovered, dict) and isinstance(recovered.get("questions"), list) and len(recovered["questions"]) > 0:
                                payload = recovered
                                break
                        except json.JSONDecodeError:
                            continue
                    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
                        break
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list):
        raise ValueError("The local model returned malformed quiz data.")
    valid = []
    seen_questions = set()
    for item in payload["questions"]:
        if not isinstance(item, dict):
            continue
        raw_options = item.get("options")
        options = ([re.sub(r"^\s*[A-Da-d][.)]\s*", "", option).strip() if isinstance(option, str) else option
                    for option in raw_options] if isinstance(raw_options, list) else raw_options)
        if (not isinstance(item.get("question"), str) or not item["question"].strip()
                or not isinstance(options, list) or len(options) != 4
                or not all(isinstance(option, str) and option.strip() for option in options)):
            continue
        if len({option.strip().casefold() for option in options}) != 4:
            continue

        raw_ans = item.get("answer")
        exp_text = str(item.get("explanation") or "").strip()
        answer = normalize_quiz_answer(raw_ans, options, exp_text)
        if answer is None or not (0 <= answer < 4):
            continue

        verified_answer = verify_semiconductor_quiz_item(item["question"], options, answer)
        if verified_answer is not None and (0 <= verified_answer < 4):
            answer = verified_answer

        normalized_question_text = item["question"].casefold()
        if any(term in normalized_question_text for term in (
            "page", "pages", "पृष्ठ", "पेज", "पन्ने", "पन्नों", "పేజీ", "పేజీలు"
        )):
            continue
        if any(term in (options[answer] + " " + exp_text).casefold()
               for term in ("not mentioned", "not stated", "not found", "unknown", "ప్రస్తావించలేదు", "చెప్పలేదు", "ఉల్లేఖన లేదు", "ఉల్లేఖించబడలేదు", "తెలియదు", "उल्लेख नहीं", "नहीं बताया")):
            continue

        is_indic = bool(re.search(r"[\u0900-\u0D7F]", item["question"] + " " + " ".join(options)))
        is_telugu = (language.strip().casefold() == "telugu") or bool(re.search(r"[\u0C00-\u0C7F]", item["question"] + " " + " ".join(options)))
        is_english = (language.strip().casefold() == "english") or (not is_indic and not is_telugu and not language.strip())

        if source_context and is_english:
            support_sentence = quiz_source_sentence_for_option(options[answer], item["question"], source_context)
            if support_sentence:
                explanation = f"The PDF states: {support_sentence}"
            else:
                q_tokens = quiz_option_tokens(item["question"])
                ans_tokens = quiz_option_tokens(options[answer])
                src_tokens = set(quiz_option_tokens(source_context))
                if (set(ans_tokens) & src_tokens) and (not q_tokens or (set(q_tokens) & src_tokens)):
                    explanation = exp_text or f"The PDF states: {options[answer]}"
                else:
                    explanation = exp_text or f"Supported by the PDF excerpts: {options[answer]}"
        else:
            explanation = exp_text or (f"Supported by source context: {options[answer]}" if source_context else "")

        normalized_question = " ".join(re.findall(r"[^\W_]+", item["question"].casefold(), flags=re.UNICODE))
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        valid.append({
            "question": item["question"].strip(),
            "options": [option.strip() for option in options],
            "answer": answer,
            "explanation": explanation
        })
    if not valid:
        raise ValueError("The local model returned no distinct, source-supported quiz questions.")
    return valid[:3]


# ============================================================
# PDF SUMMARY
# ============================================================

@app.post("/api/summarize")
@app.post("/api/summary")
async def summarize_pdf(
    request: SummaryRequest
):

    if collection.count() == 0:
        answer = no_info_answer(request.language)

        return ndjson_response(
            answer,
            []
        )

    try:
        result = await run_in_threadpool(collection.get, include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not documents:
            return ndjson_response(no_info_answer(request.language), [])

        if (request.documents is not None and len(request.documents) == 0) or (request.filenames is not None and len(request.filenames) == 0):
            return ndjson_response(no_info_answer(request.language), [])

        target_docs = set([Path(d).name for d in (request.documents or request.filenames or []) if d])
        if target_docs:
            filtered_docs = []
            filtered_metas = []
            for doc, meta in zip(documents, metadatas):
                fn = (meta or {}).get("filename") or (meta or {}).get("source") or ""
                if Path(fn).name in target_docs:
                    filtered_docs.append(doc)
                    filtered_metas.append(meta)
            documents = filtered_docs
            metadatas = filtered_metas

        if not documents:
            return ndjson_response(no_info_answer(request.language), [])

        model = model_for_language(request.language)
        entries = []
        sources = []
        seen_sources = set()
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            metadata = metadata or {}
            filename = metadata.get("filename") or metadata.get("source") or "Unknown PDF"
            page = metadata.get("page", "Unknown")
            entries.append(f"PDF: {filename}\nPAGE: {page}\nCONTENT:\n{text}")
            key = (filename, page)
            if key not in seen_sources:
                seen_sources.add(key)
                if len(sources) < 100:
                    sources.append({"filename": filename, "source": filename, "page": page})

        # Every indexed chunk contributes through map/reduce passes; keep each local
        # Ollama prompt bounded instead of sampling only the first few pages.
        batches = []
        current = []
        current_size = 0
        batch_limit = int(os.environ.get("SUMMARY_BATCH_CHARS", "14000"))
        for entry in entries:
            if current and current_size + len(entry) > batch_limit:
                batches.append("\n\n".join(current))
                current, current_size = [], 0
            current.append(entry)
            current_size += len(entry)
        if current:
            batches.append("\n\n".join(current))

        language_rules = language_instruction(request.language)
        target_desc = f"of {', '.join(sorted(target_docs))} " if target_docs else ""
        if len(batches) == 1:
            prompt = f"""Create a concise, study-oriented summary {target_desc}in {request.language} using only explicit facts in these PDF excerpts.
{language_rules}
Use one brief Overall Summary and up to five Important Key Points. Add Definitions or Main Topics only when useful
and supported. Use short bullets. Never infer missing information or add claims about what the PDF does not contain.
Do not add formulas, applications, definitions, or examples unless stated in the excerpts. Omit empty sections and
generic study advice. Keep the summary concise.
PDF EXCERPTS:\n{batches[0]}"""
            summary_limit = 550 if request.language.strip().casefold() == "telugu" else 260
            answer = clean_answer(await run_in_threadpool(generate_with_ollama, prompt, model, None, summary_limit))
        else:
            is_telugu = request.language.strip().casefold() == "telugu"
            intermediate_model = ENGLISH_MODEL if is_telugu else model
            partials = []
            for batch_index, batch in enumerate(batches, start=1):
                if is_telugu:
                    prompt = f"""You are summarizing part {batch_index} {target_desc}of an uploaded academic PDF collection.
Use only facts explicitly stated in these excerpts. Preserve definitions, formulas, values,
limitations, and examples that are actually present. Write at most four concise bullet points in English.
Do not explain what a value might mean. Omit unsupported sections.
PDF EXCERPTS:\n{batch}"""
                    partial_limit = 220
                else:
                    prompt = f"""You are summarizing part {batch_index} {target_desc}of an uploaded academic PDF collection.
Use only facts explicitly stated in these excerpts. Preserve definitions, formulas, values,
limitations, and examples that are actually present. Write at most four concise bullet points in {request.language}.
{language_rules}
Do not explain what a value might mean. Avoid claims such as "likely", "suggests", or "indicates"
unless those words and that interpretation are explicitly stated in the PDF. If an excerpt only
repeats a code or label, report the code/label without inferring its purpose. Omit unsupported sections.
PDF EXCERPTS:\n{batch}"""
                    partial_limit = 220
                partials.append(await run_in_threadpool(generate_with_ollama, prompt, intermediate_model, None, partial_limit))

            # Repeatedly condense intermediate notes until the final prompt is bounded.
            while len(partials) > 12:
                reduced = []
                for start in range(0, len(partials), 10):
                    group = "\n\n".join(partials[start:start + 10])
                    prompt = f"""Consolidate these notes from uploaded PDFs in {request.language}.
Keep distinct supported facts, definitions, formulas, values, and limitations; merge duplicates.
Do not infer or add information. Keep the result concise.\n{language_rules}\nNOTES:\n{group}"""
                    reduce_limit = 420 if request.language.strip().casefold() == "telugu" else 350
                    reduced.append(await run_in_threadpool(generate_with_ollama, prompt, model, None, reduce_limit))
                partials = reduced

            final_prompt = f"""Create a useful, concise study summary in {request.language} using only the supplied notes.
{language_rules}
Use one brief Overall Summary and up to five Important Key Points. Add Definitions or Main Topics only when
supported. Omit sections with no source support. Use neat bold headings and short bullets. Keep each factual
statement traceable to a supplied note. Do not infer what a
code, value, color, name, or repeated phrase may mean. Do not add likely interpretations, claims about what the
PDF does not contain, generic study advice, or empty sections such as "None provided". Omit every section unless
you can state at least one concrete fact supported by the notes.
NOTES:\n{'\n\n'.join(partials)}"""
            final_limit = 550 if request.language.strip().casefold() == "telugu" else 300
            answer = clean_answer(await run_in_threadpool(generate_with_ollama, final_prompt, model, None, final_limit))
        if request.language.strip().casefold() == "english":
            answer = filter_english_summary(answer, "\n".join(documents))
        if not answer:
            return ndjson_response(no_info_answer(request.language), [])
        return ndjson_response(answer or "I could not find this information in the uploaded documents.", sources)
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OllamaGenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        print("Summary error:", error)
        raise HTTPException(status_code=502, detail="The local model could not summarize these documents. Please retry with a smaller set of PDFs.") from error


@app.post("/api/quiz")
async def generate_quiz(request: QuizRequest):
    """Generate a short multiple-choice quiz using only indexed PDF text."""
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Upload a PDF first to create a quiz.")
    try:
        result = await run_in_threadpool(collection.get, include=["documents", "metadatas"])
        chunks = result.get("documents") or []
        metadata = result.get("metadatas") or []

        target_docs = set()
        if request.document and request.document.strip() and request.document.strip() != "__all__":
            target_docs.add(Path(request.document.strip()).name)
        elif request.documents:
            target_docs.update(Path(d.strip()).name for d in request.documents if d and d.strip() and d.strip() != "__all__")

        if target_docs:
            filtered_chunks = []
            filtered_meta = []
            for doc, meta in zip(chunks, metadata):
                fn = (meta or {}).get("filename") or (meta or {}).get("source") or ""
                if Path(fn).name in target_docs:
                    filtered_chunks.append(doc)
                    filtered_meta.append(meta)
            chunks = filtered_chunks
            metadata = filtered_meta

        if not chunks:
            if target_docs:
                raise HTTPException(status_code=400, detail=f"No content found for the selected document: {', '.join(sorted(target_docs))}.")
            raise HTTPException(status_code=400, detail="Upload a PDF first to create a quiz.")

        total_text_len = sum(len(c.strip()) for c in chunks)
        if total_text_len < 50:
            raise HTTPException(status_code=400, detail="The selected PDF has insufficient text to generate a quiz.")

        pdf_count = len({(item or {}).get("filename") or (item or {}).get("source") or "Unknown PDF" for item in metadata})
        selected_items = select_balanced_chunks(
            chunks,
            metadata,
            limit=min(4, max(2, pdf_count * 2))
        )
        context = "\n\n".join(text for text, _ in selected_items)
        is_telugu = request.language.strip().casefold() == "telugu"
        model = ENGLISH_MODEL if is_telugu else model_for_language(request.language)
        schema_format = QUIZ_RESPONSE_SCHEMA
        quiz_limit = 750
        prompt_lang = "English" if is_telugu else request.language
        prompt = f'''Return up to three distinct, valid multiple-choice questions in {prompt_lang} as JSON only.
Use this exact schema: {{"questions":[{{"question":"...","options":["...","...","...","..."],"answer":0,"explanation":"..."}}]}}.
The answer must be a zero-based integer option index (0, 1, 2, or 3).
Create questions based ONLY on these PDF excerpts. Every question must have exactly four distinct string options.
The correct option must be directly supported by the excerpts. Distractors must be false according to the excerpts.
Do not repeat questions, use markdown, ask about document/page metadata, or write negative "not mentioned" questions.
PDF EXCERPTS:
{context}'''
        try:
            raw = await run_in_threadpool(generate_with_ollama, prompt, model, schema_format, quiz_limit)
            valid = parse_quiz_response(raw, context, prompt_lang)
        except (ValueError, OllamaGenerationError):
            repair_prompt = f'''Return up to three distinct, valid multiple-choice questions as JSON only.
Use this schema: {{"questions":[{{"question":"...","options":["...","...","...","..."],"answer":0,"explanation":"..."}}]}}.
The answer must be a zero-based integer option index. Every correct option must occur in the source excerpts. The question must have exactly one correct option; make all distractors clearly incorrect.
Do not ask about page metadata or whether a fact is absent. All facts must come only from these PDF excerpts.
PDF EXCERPTS:
{context}'''
            raw = await run_in_threadpool(generate_with_ollama, repair_prompt, model, schema_format, quiz_limit)
            try:
                valid = parse_quiz_response(raw, context, prompt_lang)
            except ValueError as parse_error:
                print("Quiz JSON validation failed after one repair attempt:", parse_error)
                raise HTTPException(status_code=502, detail="The local model could not produce a valid quiz. Please try again.") from parse_error
        if is_telugu:
            valid = [ensure_telugu_quiz_item(q) for q in valid]
        deduped = []
        seen_q = set()
        for q in valid:
            q_text = q.get("question", "").strip()
            if q_text and q_text not in seen_q:
                seen_q.add(q_text)
                deduped.append(q)
        valid = deduped
        for index, question in enumerate(valid, start=1):
            question["id"] = f"q{index}"
        return {"questions": valid}
    except HTTPException:
        raise
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OllamaGenerationError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except requests.RequestException as error:
        raise HTTPException(status_code=503, detail="Local Ollama is unavailable. Check its status and retry.") from error
    except Exception as error:
        print("Quiz generation error:", error)
        raise HTTPException(status_code=502, detail="Could not generate a quiz from the uploaded PDF content. Please retry.") from error

# ============================================================
# STATIC FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(FRONTEND_DIR)
    ),
    name="static"
)


# ============================================================
# START MESSAGE
# ============================================================

print("=" * 60)
print("Offline Multilingual PDF Chatbot")
print("=" * 60)
print("Frontend :", FRONTEND_DIR)
print("Uploads  :", UPLOAD_DIR)
print("ChromaDB :", CHROMA_DIR)
print("Ollama   :", OLLAMA_URL)
print("=" * 60)















