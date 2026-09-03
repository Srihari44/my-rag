import os
import glob
import hashlib
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

logger = logging.getLogger(__name__)

embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",
    api_key=SecretStr(os.environ.get("OPENROUTER_API_KEY") or ""),
    base_url="https://openrouter.ai/api/v1",
    model_kwargs={"encoding_format": "float"},
    check_embedding_ctx_length=False,
)

connection = os.environ.get("POSTGRES_CONNECTION_STRING")  # Uses psycopg3!

collection_name = "my_docs"
data_dir = "./data"


_vector_store = None

def get_vector_store() -> PGVector:
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing PGVector store (collection=%s)", collection_name)
        _vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
    return _vector_store


def get_retriever():
    return get_vector_store().as_retriever(search_kwargs={"k": 3})


def _chunk_with_ids(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Generate deterministic IDs based on content + page to enable upsert
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace("\x00", "")
        content = chunk.page_content
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", 0)
        id_str = f"{source}_{page}_{content[:100]}"
        chunk.id = hashlib.md5(id_str.encode()).hexdigest()

    return chunks


def ingest_pdf(pdf_path: str) -> int:
    """Load a single PDF, chunk it, and add it to the vector store. Returns chunk count."""
    logger.info("Loading PDF: %s", pdf_path)
    docs = PyPDFLoader(pdf_path).load()
    chunks = _chunk_with_ids(docs)
    logger.info("Split %s into %d chunks", pdf_path, len(chunks))
    get_vector_store().add_documents(chunks)
    logger.info("Added %d chunks from %s to vector store", len(chunks), pdf_path)
    return len(chunks)


def load_doc_chunks():
    """Ingest every PDF in data_dir and return a retriever over the collection."""
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))
    logger.info("Found %d PDFs in %s", len(pdf_paths), data_dir)
    docs = []
    for pdf_path in pdf_paths:
        docs.extend(PyPDFLoader(pdf_path).load())

    chunks = _chunk_with_ids(docs)
    logger.info("Split %d documents into %d chunks", len(pdf_paths), len(chunks))
    get_vector_store().add_documents(chunks)
    logger.info("Added %d chunks to vector store", len(chunks))
    return get_retriever()
