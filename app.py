from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from openai import OpenAI
from tavily import TavilyClient
import os

# Initialize FastAPI
app = FastAPI()

# Load persistent ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("insurance_docs")

# OpenAI client
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tavily client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class QueryRequest(BaseModel):
    question: str
    n_results: int = 5

@app.post("/answer")
def answer_question(request: QueryRequest):

    # Step 1 — Embed the question using OpenAI (lightweight)
    embedding_response = llm.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    query_embedding = embedding_response.data[0].embedding

    # Step 2 — Retrieve relevant chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.n_results
    )

    pdf_chunks = results["documents"][0]
    pdf_sources = results["metadatas"][0]

    # Step 3 — Internet search using Tavily
    web_results = tavily.search(
        query=request.question,
        max_results=5
    )

    # Build web context
    web_context = ""
    for i, item in enumerate(web_results["results"]):
        web_context += (
            f"WEB RESULT {i+1}\n"
            f"URL: {item['url']}\n"
            f"CONTENT: {item['content']}\n\n"
        )

    # Build PDF context
    pdf_context = ""
    for i, chunk in enumerate(pdf_chunks):
        pdf_context += (
            f"PDF RESULT {i+1}\n"
            f"FILE: {pdf_sources[i]['source']}\n"
            f"CONTENT: {chunk}\n\n"
        )

    # Step 4 — Combined context
    full_context = pdf_context + "\n" + web_context

    # Step 5 — Stronger prompt for synthesis
    prompt = f"""
You are an insurance benefits expert.

Your job is to read BOTH:
1. Extracted text from PDF plan documents
2. Internet search results

Then produce a clear, natural-language answer.

### CONTEXT START ###
{full_context}
### CONTEXT END ###

### USER QUESTION ###
{request.question}
"""

    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    return {
        "question": request.question,
        "answer": answer,
        "pdf_sources": pdf_sources,
        "web_sources": web_results["results"]
    }
