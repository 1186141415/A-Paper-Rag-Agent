from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from rag_system import RAGSystem
from data_loader import load_pdfs, process_documents
from tools import TOOLS
from agent import run_agent

from logger_config import setup_logger

logger = setup_logger()

# 初始化
app = FastAPI()

rag = None  # 先不初始化


class QueryRequest(BaseModel):
    question: str


@app.on_event("startup")
def load_rag():
    global rag
    # print("🚀 Loading RAG system...")
    logger.info("Loading RAG system...")

    docs = load_pdfs("data")
    # print("docs数量:", len(docs))
    logger.info(f"docs数量: {len(docs)}")

    chunks = process_documents(docs)
    # print("chunks数量:", len(chunks))
    logger.info(f"chunks数量: {len(chunks)}")

    rag = RAGSystem(chunks)
    rag.build_index()

    print("✅ RAG ready!")
    logger.info("RAG ready!")


# API接口
@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        answer = run_agent(req.question, TOOLS, rag)
        return {
            "question": req.question,
            "answer": answer
        }
    except Exception as e:
        logger.exception("Error occurred in /ask")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/clear')
def clear_memory():
    rag.chat_history = []
    return {"message": "memory cleared"}
