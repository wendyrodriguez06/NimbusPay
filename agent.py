"""
agent.py
--------
Etapa 2 del proyecto: el agente de IA que responde preguntas en lenguaje
natural, usando como única fuente de verdad la base de conocimiento
generada por ingest.py.

Uso:
    python agent.py
"""

import os
import sys
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

VECTORSTORE_PATH = "vectorstore/faiss_index"

PROMPT_TEMPLATE = """Eres un asistente interno de la empresa. Responde la
pregunta del usuario ÚNICAMENTE con base en el siguiente contexto extraído
de los documentos internos. Si la respuesta no está en el contexto, di
claramente que no encontraste esa información en los documentos, no
inventes datos.

Contexto:
{context}

Pregunta: {question}

Respuesta clara y en lenguaje natural:"""


def load_agent():
    if not os.path.exists(VECTORSTORE_PATH):
        print("❌ No se encontró la base de conocimiento.")
        print("   Primero ejecuta: python ingest.py data/tu_documento.pdf")
        sys.exit(1)

    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ No se encontró GOOGLE_API_KEY.")
        print("   Copia .env.example como .env y coloca tu clave de Google AI Studio.")
        sys.exit(1)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain


def run_cli():
    print("🤖 Alura Agente — pregunta lo que quieras sobre tu documento.")
    print("   (escribe 'salir' para terminar)\n")

    qa_chain = load_agent()

    while True:
        question = input("Tú: ").strip()
        if question.lower() in ("salir", "exit", "quit"):
            print("👋 ¡Hasta luego!")
            break
        if not question:
            continue

        result = qa_chain.invoke({"query": question})
        print(f"\nAgente: {result['result']}\n")

        pages = sorted(
            {doc.metadata.get("page", "?") for doc in result["source_documents"]}
        )
        print(f"   📎 Fuente: página(s) {pages}\n")


if __name__ == "__main__":
    run_cli()
