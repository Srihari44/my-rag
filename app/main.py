import logging
import shutil
from pathlib import Path
from typing import Annotated, List
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

from .logging_config import configure_logging
from .loader import data_dir, ingest_pdf
from .rag import answer_query
from .response import RAGResponse

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    query: str


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int


@app.get("/health")
def health_check():
    return {"status": "Service is up"}


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    dest_dir = Path(data_dir)
    if not dest_dir.exists():
        return []
    return sorted(
        (
            DocumentInfo(filename=p.name, size_bytes=p.stat().st_size)
            for p in dest_dir.glob("*.pdf")
        ),
        key=lambda d: d.filename.lower(),
    )


@app.post("/chat", response_model=RAGResponse)
def chat(request: ChatRequest):
    logger.info("Received chat query: %r", request.query)
    try:
        response = answer_query(request.query)
    except Exception:
        logger.exception("Failed to answer query: %r", request.query)
        raise
    logger.info("Answered chat query: %r", request.query)
    return response


@app.post("/upload", response_model=UploadResponse)
def upload(file: Annotated[UploadFile, File()]):
    logger.info("Received upload: %s", file.filename)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.warning("Rejected upload with non-PDF filename: %s", file.filename)
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    dest_dir = Path(data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(file.filename).name

    with dest_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    logger.info("Saved upload to %s", dest_path)

    try:
        chunks_added = ingest_pdf(str(dest_path))
    except Exception:
        logger.exception("Failed to ingest PDF: %s", dest_path)
        raise
    logger.info("Ingested %s: %d chunks added", dest_path.name, chunks_added)
    return UploadResponse(filename=dest_path.name, chunks_added=chunks_added)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
