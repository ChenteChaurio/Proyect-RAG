"""
app.py
------
Interfaz de línea de comandos para la aplicación RAG.

Uso
---
# Primera ejecución: indexar el blog post en Pinecone
py app.py --index

# Hacer una pregunta (reutiliza el índice Pinecone existente)
py app.py --ask "¿Qué es la descomposición de tareas?"

# Sesión interactiva de preguntas
py app.py --interactive

# Indexar Y preguntar en un solo comando
py app.py --index --ask "¿Cuáles son los tipos de memoria en agentes de IA?"
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Load .env before importing anything that reads env-vars
load_dotenv()

from src.indexer import build_vector_store, load_vector_store
from src.rag import build_rag_chain, ask


def print_result(result: dict) -> None:
    """Imprime el resultado del RAG con formato legible."""
    print("\n" + "=" * 60)
    print("RESPUESTA:")
    print("=" * 60)
    print(result["answer"])

    sources = result.get("sources", [])
    if sources:
        print("\n" + "-" * 60)
        print(f"FUENTES ({len(sources)} fragmento(s) recuperado(s)):")
        print("-" * 60)
        for i, doc in enumerate(sources, 1):
            src = doc.metadata.get("source", "desconocido")
            start = doc.metadata.get("start_index", "?")
            snippet = doc.page_content[:150].replace("\n", " ")
            print(f"  [{i}] {src}  (offset {start})")
            print(f"       \"{snippet}...\"")
    print()


def interactive_session(chain) -> None:
    """Inicia una sesión interactiva de preguntas y respuestas."""
    print("\nSesión RAG iniciada. Escribe 'salir' o 'exit' para terminar.\n")
    while True:
        try:
            question = input("Pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit"}:
            print("¡Hasta luego!")
            break

        result = ask(chain, question)
        print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo RAG con LangChain + OpenAI + Pinecone"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="(Re)indexar el documento fuente en Pinecone antes de consultar.",
    )
    parser.add_argument(
        "--ask",
        metavar="PREGUNTA",
        help="Hacer una pregunta y mostrar la respuesta.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Iniciar una sesión interactiva de preguntas y respuestas.",
    )
    args = parser.parse_args()

    # Validar variables de entorno requeridas
    required_vars = ["GROQ_API_KEY", "PINECONE_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"[ERROR] Faltan variables de entorno: {', '.join(missing)}")
        print("        Copia .env.example en .env y completa los valores.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Paso 1 – Indexación (opcional)
    # -----------------------------------------------------------------------
    if args.index:
        print("[App] Indexando documento en Pinecone ...")
        vector_store = build_vector_store()
    else:
        print("[App] Conectando al índice Pinecone existente ...")
        vector_store = load_vector_store()

    # -----------------------------------------------------------------------
    # Paso 2 – Construir la cadena
    # -----------------------------------------------------------------------
    print("[App] Construyendo cadena RAG ...")
    chain = build_rag_chain(vector_store)

    # -----------------------------------------------------------------------
    # Paso 3 – Consultar
    # -----------------------------------------------------------------------
    if args.ask:
        result = ask(chain, args.ask)
        print_result(result)

    if args.interactive:
        interactive_session(chain)

    if not args.ask and not args.interactive:
        # Por defecto, iniciar modo interactivo si no se pasa ningún flag
        interactive_session(chain)


if __name__ == "__main__":
    main()
