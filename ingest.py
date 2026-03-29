import os
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Persistent ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="insurance_docs",
    metadata={"hnsw:space": "cosine"}
)

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def ingest_pdfs(pdf_folder="pdfs"):
    doc_id = 0

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            print(f"Ingesting {path}")

            text = load_pdf_text(path)
            chunks = chunk_text(text)

            embeddings = model.encode(chunks).tolist()

            ids = [f"doc_{doc_id+i}" for i in range(len(chunks))]
            doc_id += len(chunks)

            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=[{"source": filename}] * len(chunks)
            )

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_pdfs()
