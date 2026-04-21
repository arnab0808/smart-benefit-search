from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from openai import OpenAI
from tavily import TavilyClient
import os
import uuid

# Initialize FastAPI
app = FastAPI()

# Load persistent ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("insurance_docs")

# OpenAI client
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tavily client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# -------------------------
# STATEFUL MEMORY STORE
# -------------------------
conversation_store = {}   # { session_id: [ {role, content}, ... ] }

class QueryRequest(BaseModel):
    session_id: str | None = None
    question: str
    n_results: int = 5

@app.post("/answer")
def answer_question(request: QueryRequest):

    # -------------------------
    # SESSION ID HANDLING
    # -------------------------
    session_id = request.session_id or str(uuid.uuid4())

    # Initialize conversation if new session
    if session_id not in conversation_store:
        conversation_store[session_id] = [
            {"role": "system", "content": "You are an insurance benefits expert."}
        ]

    # -------------------------
    # STEP 1 — Embed the question
    # -------------------------
    embedding_response = llm.embeddings.create(
        model="text-embedding-3-small",
        input=request.question
    )
    query_embedding = embedding_response.data[0].embedding

    # -------------------------
    # STEP 2 — Retrieve from ChromaDB
    # -------------------------
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request.n_results
    )

    pdf_chunks = results["documents"][0]
    pdf_sources = results["metadatas"][0]

    # -------------------------
    # STEP 3 — Internet search via Tavily
    # -------------------------
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

    # Combined context
    full_context = pdf_context + "\n" + web_context

    # -------------------------
    # STEP 4 — Build the prompt
    # -------------------------
    prompt = f"""
You are an insurance benefits expert. You will be answering patient's question based on the patient health insurance plan. The first line of the question will consist the plan name.

Your job is to read BOTH:
1. Extracted text or image from PDF plan documents
2. Internet search results

If you see a plan name in the request which does not exist in the document or in internet, answer saying "This is an invalid plan name". If you see similar named plan, you may suggest the correct plan name.

When you get your answer from the texts extracted from pdf, no need to go for internet results.

Do not answer any question unrelated to healthcare benefits. Rather say "I am only set up to answer question related to Health Care Benefits."

If the data could not be found from the extracted text, then search internet.

### CONTEXT START ###
{full_context}
### CONTEXT END ###

### USER QUESTION ###
{request.question}
"""

    # -------------------------
    # STATEFUL MEMORY — ADD USER MESSAGE
    # -------------------------
    conversation_store[session_id].append(
        {"role": "user", "content": prompt}
    )

    # -------------------------
    # STEP 5 — LLM CALL WITH FULL HISTORY
    # -------------------------
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_store[session_id]
    )

    answer = response.choices[0].message.content

    # Add assistant reply to history
    conversation_store[session_id].append(
        {"role": "assistant", "content": answer}
    )

    # -------------------------
    # FINAL RESPONSE
    # -------------------------
    return {
        "session_id": session_id,
        "question": request.question,
        "answer": answer,
        "pdf_sources": pdf_sources,
        "web_sources": web_results["results"]
    }
