import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openrouter import ChatOpenRouter
from typing import List
from langchain_core.documents import Document
from langfuse.langchain import CallbackHandler

from .loader import get_retriever
from .response import RAGResponse

logger = logging.getLogger(__name__)

langfuse_handler = CallbackHandler()

llm = ChatOpenRouter(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", temperature=0.6
)
model = llm.with_structured_output(RAGResponse).with_retry(stop_after_attempt=2)

PROMPT = ChatPromptTemplate.from_template("""
Answer the question using only the provided context.

Do not use your general knowledge.

If the answer cannot be found in the provided context,
respond that the information is not available in the
provided documents.

Do not make assumptions or invent information.

Context:
{context}

Question: {question}

Answer (include sources):""")


def format_docs_with_sources(docs: List[Document]) -> str:
    logger.info("Retrieved %d documents for context", len(docs))
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain():
    retriever = get_retriever()
    return (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | model
    )


def answer_query(query: str) -> RAGResponse:
    logger.info("Building RAG chain for query: %r", query)
    rag_chain = build_rag_chain()
    result = rag_chain.invoke(query, config={"callbacks": [langfuse_handler]})
    logger.info("RAG chain produced a response for query: %r", query)
    return result if isinstance(result, RAGResponse) else RAGResponse.model_validate(result)

