import shutil
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

from .loader import data_dir, ingest_pdf
from .rag import answer_query
from .response import RAGResponse

app = FastAPI()


class ChatRequest(BaseModel):
    query: str


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


@app.get("/")
def health_check():
    return {"status": "Service is up"}


@app.post("/chat", response_model=RAGResponse)
def chat(request: ChatRequest):
    return answer_query(request.query)


@app.post("/upload", response_model=UploadResponse)
def upload(file: Annotated[UploadFile, File()]):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    dest_dir = Path(data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(file.filename).name

    with dest_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    chunks_added = ingest_pdf(str(dest_path))
    return UploadResponse(filename=dest_path.name, chunks_added=chunks_added)
