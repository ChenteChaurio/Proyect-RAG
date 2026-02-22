"""
indexer.py
----------
Responsable de:
  1. Cargar documentos desde una URL web.
  2. Dividirlos en fragmentos manejables.
  3. Generar embeddings y almacenarlos en Pinecone.

Uso (módulo independiente):
    python -m src.indexer
"""

import os
import time

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Suprimir advertencia de USER_AGENT al usar WebBaseLoader
os.environ.setdefault("USER_AGENT", "RAG-LangChain-Demo/1.0")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-groq-demo")
# Modelo de embeddings local (HuggingFace, sin costo ni API key)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # dimensión de all-MiniLM-L6-v2

# Documento fuente: blog post "LLM Powered Autonomous Agents" de Lilian Weng
SOURCE_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


def _get_pinecone_index(pc: Pinecone, index_name: str):
    """Crea el índice Pinecone si no existe y lo retorna."""
    if not pc.has_index(index_name):
        print(f"[Indexer] Creando índice Pinecone '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Esperar hasta que el índice esté listo
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        print(f"[Indexer] Índice '{index_name}' listo.")
    else:
        print(f"[Indexer] Usando índice existente '{index_name}'.")
    return pc.Index(index_name)


def load_and_split(url: str = SOURCE_URL) -> list:
    """Descarga la página web y la divide en fragmentos."""
    print(f"[Indexer] Cargando documento desde: {url}")

    # Solo conservar título, cabeceras y contenido HTML del post
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    print(
        f"[Indexer] {len(docs)} documento(s) cargado(s). "
        f"Total de caracteres: {sum(len(d.page_content) for d in docs):,}"
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # tamaño máximo por fragmento (caracteres)
        chunk_overlap=200,  # solapamiento entre fragmentos consecutivos
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)
    print(f"[Indexer] Dividido en {len(splits)} fragmentos.")
    return splits


def build_vector_store(
    splits: list | None = None,
    url: str = SOURCE_URL,
) -> PineconeVectorStore:
    """
    Genera embeddings de los fragmentos y los sube a Pinecone.

    Si se pasan *splits* se usan directamente; si no, se carga y divide
    el documento en *url* de forma automática.

    Retorna una instancia de ``PineconeVectorStore`` lista para consultar.
    """
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=pinecone_api_key)
    index = _get_pinecone_index(pc, INDEX_NAME)

    print("[Indexer] Cargando modelo de embeddings (HuggingFace) ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    if splits is None:
        splits = load_and_split(url)

    print("[Indexer] Subiendo documentos a Pinecone ...")
    vector_store.add_documents(documents=splits)
    print("[Indexer] Indexación completa.")

    return vector_store


def load_vector_store() -> PineconeVectorStore:
    """
    Conecta a un índice Pinecone *ya existente* sin reindexar documentos.
    Usar cuando el índice ya está poblado y solo se quiere consultar.
    """
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    return PineconeVectorStore(index=index, embedding=embeddings)


# ── bloque __main__ ──────────────────────────────────────────────────────────
    pinecone_api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return PineconeVectorStore(index=index, embedding=embeddings)


if __name__ == "__main__":
    # Prueba rápida: indexar el blog post y verificar que el store es consultable.
    from dotenv import load_dotenv
    load_dotenv()

    store = build_vector_store()
    results = store.similarity_search("¿Qué es la descomposición de tareas?", k=2)
    for i, doc in enumerate(results, 1):
        print(f"\n--- Resultado {i} ---")
        print(doc.page_content[:300])
