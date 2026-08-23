from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_doc_chunks():
    loader = PyPDFLoader("./test.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vector_store = Chroma.from_documents(
        documents=chunks,
        persist_directory="./chroma_db",
    )
    return vector_store
