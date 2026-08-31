import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",
    api_key=SecretStr(os.environ.get("OPENROUTER_API_KEY") or ""),
    base_url="https://openrouter.ai/api/v1",
    model_kwargs={"encoding_format": "float"},
    # Crucial flags to prevent schema mismatch errors with non-OpenAI models
    check_embedding_ctx_length=False,
)

connection = "postgresql+psycopg://postgres@localhost:5432/rag_dev"  # Uses psycopg3!
collection_name = "my_docs"


def load_doc_chunks():
    loader = PyPDFLoader("./test.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Generate deterministic IDs based on content + page to enable upsert
    for chunk in chunks:
        content = chunk.page_content
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", 0)
        id_str = f"{source}_{page}_{content[:100]}"
        chunk.id = hashlib.md5(id_str.encode()).hexdigest()

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    vector_store.add_documents(chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever
