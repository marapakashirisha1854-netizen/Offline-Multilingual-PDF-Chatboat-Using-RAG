import chromadb
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="pdf_documents"
)

question = "What is a PN junction diode?"

# Convert question into embedding
question_embedding = model.encode(question).tolist()

# Search similar chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

print("\nRelevant chunks:\n")

for i, document in enumerate(results["documents"][0]):
    print(f"Result {i + 1}:")
    print(document)
    print()