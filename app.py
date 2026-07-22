"""
app.py
------
Interfaz web del Alura Agente, construida con Streamlit.
Permite subir/seleccionar un PDF, procesarlo y conversar con el
agente desde el navegador, en lugar de la terminal.

Uso:
    streamlit run app.py
"""

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

DATA_DIR = Path("data")
VECTORSTORE_PATH = "vectorstore/faiss_index"

DATA_DIR.mkdir(exist_ok=True)

PROMPT_TEMPLATE = """Eres un asistente interno de la empresa. Responde la
pregunta del usuario ÚNICAMENTE con base en el siguiente contexto extraído
de los documentos internos. Si la respuesta no está en el contexto, di
claramente que no encontraste esa información en los documentos, no
inventes datos.

Contexto:
{context}

Pregunta: {question}

Respuesta clara y en lenguaje natural:"""

# ----------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Alura Agente",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

FONT_IMPORT = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html { font-size: 17px; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, .hero-title { font-family: 'Poppins', sans-serif !important; }

    h1 { font-size: 2.6rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.6rem !important; font-weight: 700 !important; }
    h3, section[data-testid="stSidebar"] h2 { font-size: 1.3rem !important; font-weight: 700 !important; }

    .subtitle { font-size: 1.15rem !important; }

    [data-testid="stChatMessageContent"] p {
        font-size: 1.15rem !important;
        line-height: 1.7 !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 1.1rem !important;
    }

    /* Texto general: párrafos, etiquetas, sidebar, selects, uploader */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    div[data-baseweb="select"] * {
        font-size: 1.08rem !important;
    }

    .stButton > button {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.1rem !important;
    }
    .source-tag {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
</style>
"""
st.markdown(FONT_IMPORT, unsafe_allow_html=True)

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

_fix_bg = "#0a0e14" if st.session_state["theme"] == "dark" else "#f4f7fc"
_fix_card = "#131c2e" if st.session_state["theme"] == "dark" else "#ffffff"
_fix_text = "#e6edf3" if st.session_state["theme"] == "dark" else "#0d1b2a"
_fix_border = "#2c5f9e" if st.session_state["theme"] == "dark" else "#1a3c6e"

FIX_UNTHEMED_AREAS = f"""
<style>
    [data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stChatInput"] {{
        background-color: {_fix_bg} !important;
        border-top: 1px solid {_fix_border} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background-color: {_fix_card} !important;
        color: {_fix_text} !important;
    }}
    [data-testid="stChatInput"] * {{ color: {_fix_text} !important; }}
    [data-testid="stChatInput"] textarea::placeholder {{ color: {_fix_text} !important; opacity: 0.55; }}

    [data-testid="stFileUploaderDropzone"] {{
        background-color: {_fix_card} !important;
        border: 1px dashed {_fix_border} !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] * {{ color: {_fix_text} !important; }}
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stBaseButton-secondary"] {{
        background-color: #1a3c6e !important;
        color: #ffffff !important;
        border: 1px solid #4d8fd1 !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: #2c5f9e !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: {_fix_card} !important;
        border-color: {_fix_border} !important;
        color: {_fix_text} !important;
    }}
    div[data-baseweb="popover"] {{ background-color: {_fix_card} !important; }}
    ul[role="listbox"] {{ background-color: {_fix_card} !important; }}
    ul[role="listbox"] li {{ color: {_fix_text} !important; }}

    /* Etiquetas de código en línea, ej: `data/` */
    code {{
        background-color: {_fix_border} !important;
        color: #ffffff !important;
        padding: 2px 8px !important;
        border-radius: 5px !important;
        font-weight: 600 !important;
    }}
</style>
"""
st.markdown(FIX_UNTHEMED_AREAS, unsafe_allow_html=True)

DARK_THEME = """
<style>
    .stApp { background-color: #0a0e14; color: #e6edf3; }
    section[data-testid="stSidebar"] {
        background-color: #0d1526;
        border-right: 1px solid #1e2a44;
    }
    section[data-testid="stSidebar"] * { color: #dbe9ff !important; }

    h1, h2, h3, h4 { color: #ffffff !important; font-family: 'Segoe UI', sans-serif; }
    .subtitle { color: #7d8fa9; font-size: 0.95rem; margin-top: -10px; }

    [data-testid="stChatMessage"] {
        background-color: #131c2e;
        border: 1px solid #1e2a44;
        border-radius: 14px;
    }
    [data-testid="stChatMessageContent"] p { color: #e6edf3 !important; }

    [data-testid="stChatInput"] {
        background-color: #131c2e !important;
        border: 1px solid #2c5f9e !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea { color: #e6edf3 !important; }

    .stButton > button {
        background-color: #1a3c6e;
        color: #ffffff;
        border: 1px solid #2c5f9e;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #2c5f9e;
        border-color: #4d8fd1;
        color: #ffffff;
    }

    .source-tag {
        display: inline-block;
        background-color: #1a3c6e;
        color: #a9c9ff;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-top: 6px;
        border: 1px solid #2c5f9e;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #131c2e;
        border: 1px dashed #2c5f9e;
    }
    div[data-baseweb="select"] > div { background-color: #131c2e; border-color: #2c5f9e; }
    hr { border-color: #1e2a44; }
</style>
"""

LIGHT_THEME = """
<style>
    .stApp { background-color: #f4f7fc; color: #0d1b2a; }
    section[data-testid="stSidebar"] {
        background-color: #e8effc;
        border-right: 1px solid #cfe0fb;
    }
    section[data-testid="stSidebar"] * { color: #0d1b2a !important; }

    h1, h2, h3, h4 { color: #0d1b2a !important; font-family: 'Segoe UI', sans-serif; }
    .subtitle { color: #4a5a72; font-size: 0.95rem; margin-top: -10px; }

    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #cfe0fb;
        border-radius: 14px;
    }

    [data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 1px solid #1a3c6e !important;
        border-radius: 12px !important;
    }

    .stButton > button {
        background-color: #1a3c6e;
        color: #ffffff;
        border: 1px solid #1a3c6e;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #2c5f9e;
        border-color: #2c5f9e;
    }

    .source-tag {
        display: inline-block;
        background-color: #dbe9ff;
        color: #1a3c6e;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-top: 6px;
        border: 1px solid #a9c9ff;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff;
        border: 1px dashed #1a3c6e;
    }
    hr { border-color: #cfe0fb; }
</style>
"""

st.markdown(DARK_THEME if st.session_state["theme"] == "dark" else LIGHT_THEME, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_paths: tuple):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        # Etiquetamos cada fragmento con el nombre del archivo de origen
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source_file"] = Path(pdf_path).name
        all_chunks.extend(chunks)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore


@st.cache_resource(show_spinner=False)
def load_existing_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )


def get_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


# ----------------------------------------------------------------------
# Barra lateral: gestión del documento
# ----------------------------------------------------------------------
with st.sidebar:
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.markdown("## 🤖 Alura Agente")
    with top_col2:
        icon = "☀️" if st.session_state["theme"] == "dark" else "🌙"
        if st.button(icon, help="Cambiar modo claro/oscuro", key="theme_toggle"):
            st.session_state["theme"] = (
                "light" if st.session_state["theme"] == "dark" else "dark"
            )
            st.rerun()

    st.markdown(
        '<p class="subtitle">Pregunta lo que quieras sobre tus documentos internos</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 📄 Documento")

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ No se encontró GOOGLE_API_KEY en tu archivo .env")

    existing_pdfs = sorted([p.name for p in DATA_DIR.glob("*.pdf")])

    uploaded_file = st.file_uploader("Sube un nuevo PDF", type=["pdf"])
    if uploaded_file is not None:
        save_path = DATA_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Guardado: {uploaded_file.name}")
        existing_pdfs = sorted([p.name for p in DATA_DIR.glob("*.pdf")])
        # Selecciona automáticamente TODOS los documentos, incluido el nuevo
        st.session_state["pdf_select"] = existing_pdfs

    if "pdf_select" not in st.session_state:
        st.session_state["pdf_select"] = existing_pdfs
    else:
        # Quita de la selección guardada archivos que ya no existen
        st.session_state["pdf_select"] = [
            p for p in st.session_state["pdf_select"] if p in existing_pdfs
        ]

    selected_pdfs = st.multiselect(
        "Documentos a incluir en la base de conocimiento",
        options=existing_pdfs,
        key="pdf_select",
        help="El agente responderá usando el contenido combinado de todos los documentos que marques aquí.",
    )

    process_btn = st.button("⚙️ Procesar documento(s)", use_container_width=True)

    if process_btn and selected_pdfs:
        with st.spinner(f"Leyendo {len(selected_pdfs)} documento(s), fragmentando y generando embeddings..."):
            build_vectorstore.clear()
            pdf_paths = tuple(sorted(str(DATA_DIR / name) for name in selected_pdfs))
            vs = build_vectorstore(pdf_paths)
            st.session_state["vectorstore"] = vs
            st.session_state["chain"] = get_qa_chain(vs)
            st.session_state["loaded_docs"] = selected_pdfs
            st.session_state["messages"] = []
        st.success(f"✅ Base de conocimiento lista con {len(selected_pdfs)} documento(s)")
    elif process_btn and not selected_pdfs:
        st.warning("Selecciona al menos un documento antes de procesar.")

    if st.session_state.get("loaded_docs"):
        st.caption("📚 Base de conocimiento actual: " + ", ".join(st.session_state["loaded_docs"]))

    st.divider()
    if st.button("🗑️ Reiniciar conversación", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ----------------------------------------------------------------------
# Cargar vectorstore existente si no hay una en sesión
# ----------------------------------------------------------------------
if "chain" not in st.session_state:
    if os.path.exists(VECTORSTORE_PATH):
        vs = load_existing_vectorstore()
        st.session_state["vectorstore"] = vs
        st.session_state["chain"] = get_qa_chain(vs)
    else:
        st.session_state["chain"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ----------------------------------------------------------------------
# Encabezado principal (banner ilustrado)
# ----------------------------------------------------------------------
banner_bg = (
    "linear-gradient(135deg, #0d1b2a 0%, #1a3c6e 55%, #2c5f9e 100%)"
    if st.session_state["theme"] == "dark"
    else "linear-gradient(135deg, #1a3c6e 0%, #2c5f9e 55%, #4d8fd1 100%)"
)

st.markdown(
    f"""
    <div style="
        background: {banner_bg};
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        display: flex;
        align-items: center;
        gap: 1.8rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 30px rgba(13, 27, 42, 0.35);
    ">
        <svg width="76" height="76" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="48" fill="rgba(255,255,255,0.12)"/>
            <rect x="30" y="34" width="40" height="32" rx="10" fill="#e6edf3"/>
            <circle cx="40" cy="50" r="5" fill="#1a3c6e"/>
            <circle cx="60" cy="50" r="5" fill="#1a3c6e"/>
            <rect x="46" y="18" width="8" height="14" rx="4" fill="#e6edf3"/>
            <circle cx="50" cy="14" r="5" fill="#a9c9ff"/>
            <path d="M22 58 Q15 58 15 65 Q15 72 22 72" stroke="#e6edf3" stroke-width="4" fill="none" stroke-linecap="round"/>
            <path d="M78 58 Q85 58 85 65 Q85 72 78 72" stroke="#e6edf3" stroke-width="4" fill="none" stroke-linecap="round"/>
        </svg>
        <div>
            <div class="hero-title" style="color:#ffffff; font-size:2.3rem; font-weight:800; line-height:1.1;">
                💬 Conversa con tus documentos
            </div>
            <div style="color:#dbe9ff; font-size:1.15rem; margin-top:6px;">
                Respuestas en lenguaje natural, basadas únicamente en tu base de conocimiento.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state["chain"] is None:
    st.info("👈 Sube o selecciona uno o varios PDF en la barra lateral y dale a **Procesar documento(s)** para comenzar.")
else:
    # Mostrar historial
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown(
                    f'<span class="source-tag">📎 {msg["sources"]}</span>',
                    unsafe_allow_html=True,
                )

    # Entrada de chat
    question = st.chat_input("Escribe tu pregunta sobre el documento...")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                result = st.session_state["chain"].invoke({"query": question})
                answer = result["result"]
                # Agrupamos página(s) por archivo de origen, ej: "brief.pdf (pág. 1, 3)"
                by_file = {}
                for doc in result["source_documents"]:
                    fname = doc.metadata.get("source_file", "documento")
                    page = doc.metadata.get("page", "?")
                    by_file.setdefault(fname, set()).add(page)
                sources = " · ".join(
                    f"{fname} (pág. {', '.join(str(p) for p in sorted(pages))})"
                    for fname, pages in by_file.items()
                )
            st.markdown(answer)
            st.markdown(
                f'<span class="source-tag">📎 {sources}</span>',
                unsafe_allow_html=True,
            )

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )