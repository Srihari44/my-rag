import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model_kwargs={"encoding_format": "float"},
    # Crucial flags to prevent schema mismatch errors with non-OpenAI models
    check_embedding_ctx_length=False,
)


def load_doc_chunks():
    loader = PyPDFLoader("./test.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = Chroma.from_documents(
        documents=chunks, persist_directory="./chroma_db", embedding=embeddings
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(chunks, k=3)
    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    )
    return retriever
