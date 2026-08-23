from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openrouter import ChatOpenRouter

from my_rag.loader import load_doc_chunks

llm = ChatOpenRouter(model="openrouter/free", temperature=0.8)


def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
    return "\n\n".join(formatted)


def main_rag():
    vectorstore = load_doc_chunks()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below. Include which sources you used. If the answer is not in the context, respond with: "I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer (include sources):""")

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    query = input("Enter your query: ")
    answer = rag_chain.invoke(query)
    print(f"Answer: {answer}")
