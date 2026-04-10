from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from rag_system import RAGSystem
from data_loader import load_pdfs, process_documents
from tools import TOOLS
from agent import run_agent
from config import DEEPSEEK_BASE_URL, EMBEDDING_BASE_URL, CHAT_MODEL, EMBEDDING_MODEL

from session_manager import SessionManager

session_manager = SessionManager(max_turns=3)

from logger_config import setup_logger

logger = setup_logger()

# 初始化
app = FastAPI()

rag = None  # 先不初始化


class QueryRequest(BaseModel):
    session_id: str
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

    logger.info(f"DEEPSEEK_BASE_URL={DEEPSEEK_BASE_URL}")
    logger.info(f"EMBEDDING_BASE_URL={EMBEDDING_BASE_URL}")
    logger.info(f"CHAT_MODEL={CHAT_MODEL}")
    logger.info(f"EMBEDDING_MODEL={EMBEDDING_MODEL}")

    rag = RAGSystem(chunks)
    rag.build_index()

    print("✅ RAG ready!")
    logger.info("RAG ready!")


# API接口
@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        history = session_manager.get_history(req.session_id)

        answer = run_agent(
            req.question,
            TOOLS,
            rag=rag,
            chat_history=history
        )

        session_manager.append_turn(
            req.session_id,
            req.question,
            answer
        )

        return {
            "session_id": req.session_id,
            "question": req.question,
            "answer": answer
        }
    except Exception as e:
        logger.exception("Error occurred in /ask")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/clear/{session_id')
def clear_session(session_id: str):
    session_manager.clear_session(session_id)
    return {
        "session_id": session_id,
        "message": "session cleared"
    }
