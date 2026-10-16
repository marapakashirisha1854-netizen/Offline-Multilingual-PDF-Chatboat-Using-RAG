import chromadb
from sentence_transformers import SentenceTransformer
import requests

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="pdf_documents"
)


def ask_question(question):

    # Convert question into embedding
    question_embedding = model.encode(question).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = ""

    for i, document in enumerate(documents):
        page = metadatas[i]["page"]
        filename = metadatas[i]["filename"]

        context += f"\nSource: {filename}, Page: {page}\n"
        context += document + "\n"

    # Prompt for the local LLM
    prompt = f"""
You are a PDF question-answering assistant.

Answer the question using ONLY the information provided in the PDF context.

If the answer is not present in the context, say:
"I could not find this information in the uploaded documents."

Keep the answer short and clear.

PDF Context:
{context}

Question:
{question}

Answer:
"""

    # Send prompt to Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"]

    return answer, metadatas


question = input("\nAsk a question about your PDF: ")

answer, sources = ask_question(question)

print("\nAnswer:")
print(answer)

print("\nSources:")

for source in sources:
    print(
        f"- {source['filename']} | Page {source['page']}"
    )