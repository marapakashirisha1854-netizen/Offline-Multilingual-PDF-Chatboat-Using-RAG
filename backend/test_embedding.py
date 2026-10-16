from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "PN junction diode is formed by P-type and N-type semiconductors."

embedding = model.encode(text)

print("Embedding created successfully!")
print("Vector size:", len(embedding))
print("First 5 values:", embedding[:5])