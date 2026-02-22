"""
rag.py
------
Implementa el pipeline RAG (Generación con Recuperación Aumentada) usando:
  - Pinecone         → base de datos vectorial / retriever
  - Groq             → LLM (llama-3.3-70b-versatile)
  - HuggingFace      → embeddings locales (all-MiniLM-L6-v2, sin API key)
  - LangChain LCEL   → cadena declarativa con soporte de citas

Se exponen dos funciones públicas:
  - build_rag_chain(vector_store)  →  retorna una cadena LCEL
  - ask(chain, question)           →  retorna {"answer": ..., "sources": [...]}
"""

import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_pinecone import PineconeVectorStore

# ---------------------------------------------------------------------------
# Plantilla del prompt
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """Eres un asistente experto que responde preguntas sobre \
agentes de IA, planificación, memoria y uso de herramientas, basándote \
exclusivamente en posts de investigación.

Usa ÚNICAMENTE el contexto siguiente para responder. Si el contexto no \
contiene suficiente información, responde: \
"No lo sé con base en el contexto proporcionado."

Contexto:
{context}
"""

RAG_HUMAN_PROMPT = "{question}"


def _format_docs(docs: list) -> str:
    """Concatena el contenido de los documentos para inyectarlo en el prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store: PineconeVectorStore, k: int = 4):
    """
    Construye una cadena RAG usando LangChain LCEL.

    Parámetros
    ----------
    vector_store : PineconeVectorStore
        Vector store de Pinecone ya poblado con documentos.
    k : int
        Número de fragmentos a recuperar por consulta.

    Retorna
    -------
    chain
        Un runnable LCEL que acepta ``{"question": str}`` y retorna
        ``{"answer": str, "sources": list[Document]}``.
    """
    llm_model = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=llm_model, temperature=0)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT),
        ]
    )

    # ------------------------------------------------------------------
    # Cadena LCEL
    # ------------------------------------------------------------------
    # Paso 1: recuperar documentos (guardados para citas)
    # Paso 2: formatear docs → string de contexto
    # Paso 3: ejecutar prompt + LLM + parser
    # Paso 4: retornar respuesta + documentos fuente originales
    # ------------------------------------------------------------------

    retrieve_docs = RunnableLambda(lambda x: retriever.invoke(x["question"]))

    chain = (
        RunnablePassthrough.assign(docs=retrieve_docs)
        | RunnablePassthrough.assign(
            context=lambda x: _format_docs(x["docs"]),
        )
        | RunnablePassthrough.assign(
            answer=(
                prompt
                | llm
                | StrOutputParser()
            )
        )
        | RunnableLambda(
            lambda x: {
                "answer": x["answer"],
                "sources": x["docs"],
            }
        )
    )

    return chain


def ask(chain, question: str) -> dict:
    """
    Wrapper de conveniencia que invoca la cadena y retorna el resultado.

    Parámetros
    ----------
    chain
        La cadena retornada por ``build_rag_chain``.
    question : str
        La pregunta del usuario.

    Retorna
    -------
    dict
        ``{"answer": str, "sources": list[Document]}``
    """
    return chain.invoke({"question": question})
