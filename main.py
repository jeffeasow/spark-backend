from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import load_vectorstore, build_spark
import os

app = FastAPI()

# Allow requests from your Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Spark once when the server starts
print("Loading Spark...")
vectorstore = load_vectorstore()
spark_chain = build_spark(vectorstore)
print("Spark ready.")

class ChatRequest(BaseModel):
    message: str
    fingerprint: str = ""

class MirrorRequest(BaseModel):
    samples: str

@app.get("/")
def health():
    return {"status": "Spark is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    query = req.message
    if req.fingerprint:
        query = f"[User style: {req.fingerprint[:200]}]\n\n{req.message}"
    response = spark_chain.invoke(query)
    return {"response": response}

@app.post("/mirror")
def mirror(req: MirrorRequest):
    prompt = f"""
Analyse these writing samples from an Indian corporate professional.
Identify their communication style concisely:

WARMTH: [high/medium/low] — [one sentence]
DIRECTNESS: [high/medium/low] — [one sentence]
CONFIDENCE: [high/medium/low] — [one sentence]
STRENGTH: [one key strength]
GROWTH AREA: [one specific improvement]
SUMMARY: [one sentence about their unique voice]

Samples:
{req.samples}
"""
    result = spark_chain.invoke(prompt)
    return {"analysis": result}