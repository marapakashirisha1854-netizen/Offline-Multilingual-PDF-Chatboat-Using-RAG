# Local-only model and OCR setup

Run setup once while internet access is available. Normal application use reads only local model files and sends no document or audio to a cloud API. The backend uses `qwen2.5:1.5b` for English and `qwen2.5:3b` for Telugu and Hindi.

## Python dependencies

From PowerShell in this project folder:

If the existing `venv` folder is present, keep it and install/update dependencies in place:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create/activate a venv only on a machine that does not already have one:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Requirements include FastAPI/Uvicorn, pypdf, `cryptography` (for AES-encrypted PDFs), ChromaDB, SentenceTransformers, requests, PyMuPDF, Pillow, pytesseract, python-multipart, and faster-whisper. Verify the encryption dependency with:

```powershell
.\venv\Scripts\python.exe -c "import cryptography; print(cryptography.__version__)"
```

## Embedding model cache

The backend loads `paraphrase-multilingual-MiniLM-L12-v2` in local-only mode. Cache it once:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

## Offline speech model

The default is the multilingual faster-whisper `small` model (`Systran/faster-whisper-small`, converted CTranslate2 weights), which supports English, Telugu, and Hindi. `small` is the multilingual model, not `small.en`. With internet available for this one-time download, run this exact command from the project root:

```powershell
.\venv\Scripts\python.exe -c "from faster_whisper.utils import download_model; print(download_model('small', output_dir='models/faster-whisper-small'))"
```

This populates `models/faster-whisper-small` under the project root. At runtime the API loads only that local directory (it does not resolve a remote model name or download weights). To store it elsewhere, set `WHISPER_MODEL_PATH` to the local model directory before starting the backend. If the model directory is absent, `/api/transcribe` returns a setup error and does not attempt a download. The first transcription loads the model into memory; subsequent recordings reuse it.

The microphone workflow is browser `MediaRecorder` → multipart POST to local `POST /api/transcribe` → faster-whisper on CPU (`int8`) → transcription text is inserted into the question input. Explicit English, Telugu, or Hindi selections pass `en`, `te`, or `hi` as Whisper's language hint. Auto passes no language hint and lets Whisper detect the language. The browser does not use `SpeechRecognition`/`webkitSpeechRecognition` or a cloud speech service.

## Tesseract OCR (Windows)

`pytesseract` is only a Python wrapper; the Tesseract Windows application must also be installed. Tesseract is installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (version 5.5.3.20260724) with English (`eng`), Hindi (`hin`), and Telugu (`tel`) trained data. The backend detects this standard Windows path automatically, even if its process did not inherit the updated Path. You can override it with `TESSERACT_CMD`.

Use the Windows installer linked by the [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html) (UB Mannheim Windows installer). In its component selection, include English and the available Telugu (`tel`) and Hindi (`hin`) trained data. Alternatively, install with winget where available:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

Open a new PowerShell window after setup. Add `C:\Program Files\Tesseract-OCR` to the Windows user/system `Path` using Environment Variables, or set it for the current PowerShell session and verify:

```powershell
$env:Path += ';C:\Program Files\Tesseract-OCR'
tesseract --version
tesseract --list-langs
```

The language list should include `eng`; add `tel` and `hin` if those packs are available for the installed build. The upstream docs explain installing additional `.traineddata` into Tesseract's `tessdata` directory. If Tesseract is installed in a different location or is not on `Path`, configure the executable for the backend session:

```powershell
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
.\venv\Scripts\python.exe -c "import pytesseract; pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'; print(pytesseract.get_tesseract_version()); print(pytesseract.get_languages(config=''))"
```

The backend uses pypdf first and invokes Tesseract only on pages with insufficient extracted text. Normal text PDFs continue to work without Tesseract. To check OCR from the project, upload a scanned PDF through the existing upload card after `tesseract --list-langs` shows its language data; the returned chunk metadata records `extraction_method` as `OCR`.

## Ollama models and start command

Install Ollama for Windows and leave its local service running. Check installed models and pull only any missing configured model:

```powershell
ollama list
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

The backend health endpoint reports Ollama connection/model status using its local `/api/tags` endpoint. English selects `qwen2.5:1.5b`; Telugu and Hindi select `qwen2.5:3b` for Q&A, summaries, and quizzes. Default request timeouts are 4 seconds to connect and 240 seconds for generation (allowing slower local model startup); override with `OLLAMA_CONNECT_TIMEOUT` and `OLLAMA_READ_TIMEOUT` if needed.

## Offline voice

The multilingual `Systran/faster-whisper-small` model is loaded from `models/faster-whisper-small`; there is no runtime model download. If it is missing on another machine, download it once from the project root while online:

```powershell
.\venv\Scripts\python.exe -c "from faster_whisper.utils import download_model; print(download_model('small', output_dir='models/faster-whisper-small'))"
```

The microphone captures through browser `MediaRecorder`, applies supported browser automatic gain/noise suppression, and posts audio to local `/api/transcribe`. Faster-whisper runs locally on CPU (`int8`), with selected `en`, `te`, or `hi` language guidance; Auto uses model language detection. Low-volume enhancement is best-effort: microphone distance, room noise, and device drivers still affect recognition.

## Offline Telugu Text-to-Speech (TTS)

English answers continue to play through the browser's speech synthesis engine using locally installed Windows system voices (e.g. Microsoft Ravi, Heera, David, Zira), with instant playback and zero model overhead.

Because Windows does not provide offline Telugu TTS voice packages in Narrator/Speech settings, Telugu text-to-speech uses a dedicated offline VITS engine (`facebook/mms-tts-tel`) running directly in the local backend on CPU via PyTorch and Transformers. No external cloud APIs are contacted and no data leaves the computer.

- **Required Disk Space:** ~146 MB
- **Local Directory:** `models/mms-tts-tel`
- **Prerequisites:** `torch`, `transformers`, `scipy` (already included in `requirements.txt`)

To set up or refresh the offline Telugu voice model on another machine:

```powershell
.\venv\Scripts\python.exe -c "from transformers import VitsModel, AutoTokenizer; m='facebook/mms-tts-tel'; VitsModel.from_pretrained(m).save_pretrained('models/mms-tts-tel'); AutoTokenizer.from_pretrained(m).save_pretrained('models/mms-tts-tel')"
```

At runtime, the backend loads strictly from `models/mms-tts-tel` with `local_files_only=True`. The frontend automatically routes Telugu text to `POST /api/tts` to stream native Telugu WAV audio, while routing English text to browser speech synthesis.

## Start and open the application

From the project root:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in the browser. Keep the local Ollama service running for Q&A, Summary, and Quiz.

## Troubleshooting

- `cryptography` / AES error: run the pip install command above, verify the version, and restart the backend. Password-protected PDFs still require the password to be removed because this UI has no password-entry field.
- OCR setup error: run `tesseract --version` and `tesseract --list-langs`; use `TESSERACT_CMD` if the executable is outside `Path`.
- Voice model missing: run the faster-whisper download command once; restart the backend and check `/api/health` for `speech_model: ready`.
- Ollama unavailable/model missing: verify `http://127.0.0.1:11434/api/tags`, start Ollama, and pull the reported missing model.
- A lengthy summary or local generation timeout: retry after the model is loaded; the backend summarizes indexed content in bounded batches. The UI and backend remain responsive while local generation runs.

