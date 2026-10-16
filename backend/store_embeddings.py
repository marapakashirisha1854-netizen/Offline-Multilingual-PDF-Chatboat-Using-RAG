import chromadb
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="pdf_documents"
)

text_chunks = [
    "PN junction diode is formed by P-type and N-type semiconductors.",
    "A diode allows current to flow mainly in one direction.",
    "Forward bias is the condition in which a diode conducts current."
]

embeddings = model.encode(text_chunks).tolist()

collection.add(
    ids=["chunk_1", "chunk_2", "chunk_3"],
    documents=text_chunks,
    embeddings=embeddings
)

print("Embeddings stored successfully!")
print("Total documents:", collection.count())