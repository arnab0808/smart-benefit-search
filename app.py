from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import chromadb
from openai import OpenAI
from tavily import TavilyClient
import os
import uuid
import traceback

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
conversation_store = {}
MAX_TURNS = 10

class QueryRequest(BaseModel):
    session_id: str | None = None
    question: str
    n_results: int = 5


# -------------------------
# GLOBAL EXCEPTION HANDLERS
# -------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print("UNEXPECTED ERROR:", traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred. Please try again later.",
            "details": str(exc)
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


# -------------------------
# MAIN ENDPOINT
# -------------------------

@app.post("/answer")
def answer_question(request: QueryRequest):

    try:
        # -------------------------
        # SESSION ID HANDLING
        # -------------------------
        session_id = request.session_id or str(uuid.uuid4())

        if session_id not in conversation_store:
            conversation_store[session_id] = [
                {
                    "role": "system",
                    "content": (
                        "You are an insurance benefits expert. You answer questions about "
                        "health insurance plans and benefits. The first line of the first "
                        "question in a conversation will contain the plan name."
                        "Please analyze the question to find the indication of plan name."
                        "If the plan name is incorrect suggest an intended plan name in the answer."
                        "If user asks a question which is not related to Health Care Benefits, answer saying I am setup to only answer benefit questions."
                        "In follow-up questions, the user may refer to 'this plan'; infer the plan name "
                        "from earlier turns.\n\n"
                        "Your job is to:\n"
                        "1. Read extracted text from PDF plan documents.\n"
                        "2. Use internet search results only if PDF data is insufficient.\n\n"
                        "If the plan name does not exist in documents or online, answer: "
                        "'This is an invalid plan name'. If a similar plan exists, suggest it.\n\n"
                        "Do NOT answer questions unrelated to healthcare benefits. Instead say: "
                        "'I am only set up to answer question related to Health Care Benefits.'\n\n"
                        "If PDF data is insufficient, then use internet results."
                    )
                }
            ]

        # -------------------------
        # TURN LIMIT CHECK
        # -------------------------
        turn_count = sum(
            1 for msg in conversation_store[session_id]
            if msg["role"] in ["user", "assistant"]
        )

        if turn_count >= MAX_TURNS:
            return {
                "session_id": session_id,
                "question": request.question,
                "answer": "The conversation is becoming too long. Please start over.",
                "pdf_sources": [],
                "web_sources": []
            }

        # -------------------------
        # STEP 1 — Embed the question
        # -------------------------
        try:
            embedding_response = llm.embeddings.create(
                model="text-embedding-3-small",
                input=request.question
            )
            query_embedding = embedding_response.data[0].embedding
        except Exception as e:
            print("Embedding error:", traceback.format_exc())
            raise HTTPException(status_code=500, detail="Failed to generate embeddings.")

        # -------------------------
        # STEP 2 — Retrieve from ChromaDB
        # -------------------------
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=request.n_results
            )
            pdf_chunks = results["documents"][0]
            pdf_sources = results["metadatas"][0]
        except Exception as e:
            print("ChromaDB error:", traceback.format_exc())
            raise HTTPException(status_code=500, detail="Failed to retrieve data from ChromaDB.")

        # -------------------------
        # STEP 3 — Internet search via Tavily
        # -------------------------
        try:
            web_results = tavily.search(
                query=request.question,
                max_results=5
            )
        except Exception as e:
            print("Tavily error:", traceback.format_exc())
            web_results = {"results": []}  # graceful fallback

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

        full_context = pdf_context + "\n" + web_context

        # -------------------------
        # STATEFUL MEMORY — ADD USER QUESTION + CONTEXT
        # -------------------------
        conversation_store[session_id].append(
            {"role": "user", "content": request.question}
        )

        conversation_store[session_id].append(
            {"role": "system", "content": f"CONTEXT FOR THE CURRENT QUESTION:\n{full_context}"}
        )

        # -------------------------
        # STEP 4 — LLM CALL
        # -------------------------
        try:
            response = llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_store[session_id]
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print("LLM error:", traceback.format_exc())
            raise HTTPException(status_code=500, detail="Failed to generate LLM response.")

        # Add assistant reply
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

    except Exception as e:
        print("Unexpected error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error.")
