"""
ingest.py
---------
Etapa 1 del proyecto: lee el documento (PDF), lo divide en fragmentos
y crea una base de conocimiento vectorial (FAISS) que el agente usará
para responder preguntas.

Uso:
    python ingest.py data/mi_documento.pdf
"""

import sys
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

VECTORSTORE_PATH = "vectorstore/faiss_index"


def ingest_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"❌ No se encontró el archivo: {pdf_path}")
        sys.exit(1)

    print(f"📄 Leyendo documento: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   → {len(documents)} páginas cargadas.")

    print("✂️  Dividiendo el documento en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    print(f"   → {len(chunks)} fragmentos generados.")

    print("🧠 Generando embeddings con Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("💾 Creando la base vectorial (FAISS)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"✅ Base de conocimiento creada en: {VECTORSTORE_PATH}")
    print("   Ahora puedes ejecutar: python agent.py")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python ingest.py <ruta_al_pdf>")
        print("Ejemplo: python ingest.py data/manual_politicas.pdf")
        sys.exit(1)

    ingest_pdf(sys.argv[1])
