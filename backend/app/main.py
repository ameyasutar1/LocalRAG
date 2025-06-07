import os
import shutil
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from langchain_ollama import OllamaLLM
from app.db.session import get_db, init_db
from app.db.models import File as DBFile
from app.error_handlers import (
    http_exception_handler,
    sqlalchemy_exception_handler,
    generic_exception_handler,
)
from app.rag.pipeline import RAGPipeline, SUPPORTED_EXTENSIONS
from app.schemas import (
    FileUploadResponse,
    FileListItem,
    ChatResponse,
    AdminClearAllResponse,
    ChatRequest,
)
from app.services.admin_service import clear_all_service
from app.services.file_service import (
    upload_file as upload_file_service,
    list_files as list_files_service,
    delete_file as delete_file_service,
)
from app.services.chat_service import chat_service
from app.log_utils import safe_log_gotcha

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "../../data/files")
CHROMA_PATH = os.path.join(BASE_DIR, "./data/chroma_db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)
ADMIN_TOKEN = os.environ.get("CHAT_RAG_ADMIN_TOKEN", "supersecret")

# Initialize application
app = FastAPI(title="Local Rag App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Initialize database and RAG pipeline
init_db()
ollama_llm = OllamaLLM(model="gemma3:1b")
rag_pipeline = RAGPipeline(vector_db_path=CHROMA_PATH)

@app.post(
    "/api/admin/clear_all", response_model=AdminClearAllResponse
)
def clear_all(
    admin_token: str = Header(..., alias="admin-token", min_length=8, max_length=128),
    db: Session = Depends(get_db),
) -> AdminClearAllResponse:
    """
    Delete ALL files and chat history from DB and vectorstore.
    """
    if not (admin_token.isalnum() or '-' in admin_token or '_' in admin_token):
        raise HTTPException(status_code=422, detail="Invalid admin token format.")

    result = clear_all_service(
        admin_token=admin_token,
        db=db,
        rag_pipeline=rag_pipeline,
        admin_env_token=ADMIN_TOKEN,
    )
    return AdminClearAllResponse(**result)

@app.get("/api/health")
def health_check():
    """
    Check health of DB, vectorstore, and LLM.
    """
    # Database health
    try:
        session = next(get_db())
        session.execute(text("SELECT 1"))
        session.close()
        db_status = {"ok": True, "msg": "OK"}
    except Exception as e:
        db_status = {"ok": False, "msg": str(e)}

    # Vectorstore health
    try:
        _ = rag_pipeline.vectorstore.get()
        vector_status = {"ok": True, "msg": "OK"}
    except Exception as e:
        vector_status = {"ok": False, "msg": str(e)}

    # LLM health
    llm_model = getattr(ollama_llm, 'model', None) or ollama_llm.__dict__.get('model')
    try:
        _ = ollama_llm.invoke("Health check?")
        llm_status = {"ok": True, "msg": "OK", "model": llm_model}
    except Exception as e:
        llm_status = {"ok": False, "msg": str(e), "model": llm_model}

    status = "ok" if all([db_status['ok'], vector_status['ok'], llm_status['ok']]) else "degraded"
    return {"status": status, "db": db_status, "vectorstore": vector_status, "llm": llm_status}

@app.post(
    "/api/upload", response_model=FileUploadResponse
)
def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> FileUploadResponse:
    """
    Upload and ingest file into RAG pipeline.
    """
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {ext}")

    db_file = upload_file_service(file=file, db=db)
    try:
        rag_pipeline.ingest(
            db_file.filepath,
            metadata={"file_id": db_file.id, "filename": db_file.filename},
        )
    except Exception as e:
        db.delete(db_file)
        db.commit()
        os.remove(db_file.filepath)
        raise HTTPException(status_code=500, detail=f"RAG ingestion failed: {e}")

    return FileUploadResponse(id=db_file.id, filename=db_file.filename)

@app.get(
    "/api/files", response_model=list[FileListItem]
)
def list_files(db: Session = Depends(get_db)) -> list[FileListItem]:
    """
    List all uploaded files.
    """
    files = list_files_service(db=db)
    return [
        FileListItem(
            id=f.id,
            filename=f.filename,
            upload_time=f.upload_time,
            file_metadata=f.file_metadata,
        ) for f in files
    ]

@app.delete("/api/files/{file_id}")
def delete_file(
    file_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """
    Delete file from DB, disk, and vectorstore.
    """
    # First remove from DB and disk
    result = delete_file_service(file_id=file_id, db=db)
    warnings = result.get("warnings", [])

    # Load filename for vectorstore deletion
    db_file = db.query(DBFile).filter(DBFile.id == file_id).first()
    filename = db_file.filename if db_file else None

    # Attempt vectorstore deletions
    delete_errors = []
    if filename:
        for key, value in [("file_id", file_id), ("filename", filename)]:
            try:
                rag_pipeline.vectorstore.delete(where={key: value})
            except Exception as e:
                delete_errors.append(f"Vectorstore delete by {key} error: {e}")
                safe_log_gotcha(f"[DeleteFile] Chromadb delete error ({key}): {e}")

    # Reinitialize vectorstore to refresh state
    rag_pipeline.vectorstore = RAGPipeline(vector_db_path=rag_pipeline.vector_db_path).vectorstore

    return {"status": "deleted", "warnings": warnings + delete_errors}

@app.post(
    "/api/chat", response_model=ChatResponse
)
def chat(
    chat_req: ChatRequest = None,
    question: str = Query(None, min_length=3, max_length=500),
    file_id: int = Query(None),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Chat endpoint: hybrid retrieval with optional legacy query params.
    """
    req = chat_req or ChatRequest(question=question, file_id=file_id)
    response = chat_service(
        question=req.question,
        file_id=req.file_id,
        db=db,
        rag_pipeline=rag_pipeline,
        ollama_llm=ollama_llm,
        keywords=req.keywords,
        metadata_filter=req.metadata_filter,
        k=req.k,
    )
    return ChatResponse(**response)