# 🤖 Alura Agente

Agente de inteligencia artificial que responde preguntas en lenguaje natural
sobre documentos internos de una empresa (PDF), usando una arquitectura de
**RAG (Retrieval Augmented Generation)**.

## 🧱 Arquitectura

```
 ┌──────────────┐     ┌────────────────┐     ┌───────────────────┐
 │  Documento    │ →   │  ingest.py     │ →   │  Base vectorial     │
 │  PDF          │     │ (PyPDF +       │     │  FAISS (local)      │
 │               │     │  LangChain)    │     │                     │
 └──────────────┘     └────────────────┘     └──────────┬──────────┘
                                                          │
                        Pregunta del usuario              │
                               │                           ▼
                               ▼                  ┌───────────────────┐
                        ┌────────────┐            │  Retriever         │
                        │ agent.py   │ ←────────  │  (busca fragmentos │
                        │            │            │  relevantes)       │
                        └─────┬──────┘            └───────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Gemini (LLM)   │
                     │  genera la      │
                     │  respuesta      │
                     └─────────────────┘
```

**Flujo:**
1. `ingest.py` carga el PDF, lo divide en fragmentos (~1000 caracteres) y
   genera embeddings con el modelo `embedding-001` de Google.
2. Los embeddings se guardan en un índice **FAISS** local
   (`vectorstore/faiss_index`).
3. `agent.py` levanta un chat por consola: cada pregunta se compara contra
   el índice vectorial, se recuperan los fragmentos más relevantes y se
   envían junto con la pregunta al modelo **Gemini 1.5 Flash**, que genera
   la respuesta final en lenguaje natural.

## 🛠️ Tecnologías

- Python 3.10+
- LangChain
- PyPDF
- FAISS (base vectorial local)
- Google Gemini (`gemini-1.5-flash` + `embedding-001`)

## 🚀 Cómo ejecutarlo

1. **Clonar el repositorio y crear un entorno virtual**
   ```bash
   git clone <url-de-tu-repo>
   cd alura-agente
   python -m venv venv
   source venv/bin/activate   # En Windows: venv\Scripts\activate
   ```

2. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar la clave de API**
   ```bash
   cp .env.example .env
   ```
   Edita `.env` y coloca tu clave de [Google AI Studio](https://aistudio.google.com/app/apikey)
   (es gratuita).

4. **Colocar tu documento**
   Copia tu PDF dentro de la carpeta `data/`, por ejemplo:
   `data/manual_politicas.pdf`

5. **Procesar el documento (Etapa 1)**
   ```bash
   python ingest.py data/manual_politicas.pdf
   ```

6. **Conversar con el agente (Etapa 2)**
   ```bash
   python agent.py
   ```

## 💬 Ejemplos de preguntas y respuestas

> **Pregunta:** ¿Cuántos días de vacaciones corresponden por año según la política interna?
> **Agente:** Según el documento, los colaboradores tienen derecho a 15 días hábiles de vacaciones por año trabajado. 📎 Fuente: página 4

> **Pregunta:** ¿Cuál es el procedimiento para solicitar reembolsos de gastos?
> **Agente:** El documento indica que los reembolsos deben solicitarse dentro de los 30 días posteriores al gasto, adjuntando la factura original al formulario del área de Finanzas. 📎 Fuente: página 7

> **Pregunta:** ¿El documento menciona la política de vacaciones en Marte?
> **Agente:** No encontré esa información en los documentos proporcionados.

*(Reemplaza estos ejemplos por preguntas y respuestas reales de tu propio documento antes de entregar el proyecto).*

## ☁️ Despliegue en OCI (Etapa 3)

> Pendiente de completar tras el deploy: agregar aquí el enlace público o
> una captura de pantalla de la aplicación corriendo en Oracle Cloud
> Infrastructure (OCI Compute).

## 📁 Estructura del proyecto

```
alura-agente/
├── data/                  # Documentos PDF de entrada
├── vectorstore/           # Índice FAISS generado (no se sube a git)
├── ingest.py              # Etapa 1: procesamiento del documento
├── agent.py                # Etapa 2: agente conversacional
├── requirements.txt
├── .env.example
└── README.md
```
