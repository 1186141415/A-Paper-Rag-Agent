from fastapi import FastAPI
from pydantic import BaseModel

from rag_system import RAGSystem
from data_loader import load_pdfs,process_documents
from tools import TOOLS
from agent import run_agent


# 初始化
app = FastAPI()

rag = None  # 先不初始化


class QueryRequest(BaseModel):
    question: str


@app.on_event("startup")
def load_rag():
    global rag
    print("🚀 Loading RAG system...")

    docs = load_pdfs("data")
    print("docs数量:", len(docs))

    chunks = process_documents(docs)
    print("chunks数量:", len(chunks))

    rag = RAGSystem(chunks)
    rag.build_index()

    print("✅ RAG ready!")


# API接口
@app.post("/ask")
def ask(req: QueryRequest):
    answer = run_agent(req.question, TOOLS, rag)
    return {
        "question": req.question,
        "answer": answer
    }


@app.post('/clear')
def clear_memory():
    rag.chat_history = []
    return {"message": "memory cleared"}
