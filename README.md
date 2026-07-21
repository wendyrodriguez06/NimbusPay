Alura Agente — Asistente de Documentos con IA

Agente de inteligencia artificial que responde preguntas en lenguaje natural sobre documentos internos de una empresa, usando una arquitectura de RAG (Retrieval Augmented Generation) sobre el modelo Gemini de Google.

Proyecto final del programa de IA de Alura — convierte documentos PDF estáticos (manuales, políticas, informes) en una fuente de conocimiento conversacional, accesible por cualquier persona del equipo sin necesidad de abrir o buscar dentro de los archivos.

📌 Tabla de contenidos
¿Qué problema resuelve?
Arquitectura
Cómo funciona el agente, paso a paso
Tecnologías
Estructura del proyecto
Cómo ejecutarlo localmente
Interfaz web (Streamlit)
Ejemplos de preguntas y respuestas
Despliegue en OCI
Limitaciones conocidas
🎯 ¿Qué problema resuelve?

Los equipos internos de una empresa (fintech, consultora o startup) suelen tener grandes volúmenes de documentos —manuales, políticas, informes, hojas de cálculo— y las personas colaboradoras pierden tiempo valioso buscando información específica dentro de ellos.

Alura Agente resuelve esto exponiendo el contenido de esos documentos a través de un chat: cualquier persona puede preguntar en español natural ("¿cuántos días de vacaciones tengo?", "¿qué lenguaje usa el backend?") y recibir una respuesta directa, sin abrir un solo PDF.

🧱 Arquitectura
 ┌───────────────┐     ┌────────────────┐     ┌──────────────────────┐
 │   Documento    │ →   │   ingest.py    │ →   │    Base vectorial     │
 │   PDF          │     │  (PyPDF +      │     │    FAISS (local)      │
 │                │     │   LangChain)   │     │                       │
 └───────────────┘     └────────────────┘     └───────────┬───────────┘
                                                            │
                          Pregunta del usuario              │
                                 │                          ▼
                                 ▼                 ┌──────────────────┐
                          ┌─────────────┐          │    Retriever      │
                          │  agent.py   │ ←──────  │  (busca los k      │
                          │  / app.py   │          │  fragmentos más    │
                          └──────┬──────┘          │  relevantes)       │
                                 │                  └──────────────────┘
                                 ▼
                        ┌──────────────────┐
                        │   Gemini (LLM)    │
                        │   genera la       │
                        │   respuesta final │
                        └──────────────────┘

El proyecto sigue el patrón RAG: en lugar de "entrenar" un modelo con los documentos (costoso y lento), se indexan sus fragmentos como vectores y, en el momento de cada pregunta, se recuperan solo los fragmentos relevantes para dárselos al modelo de lenguaje como contexto. Esto permite que el agente responda con información actualizada y verificable, citando la página exacta de donde salió cada respuesta, y sin inventar datos que no estén en el documento.

⚙️ Cómo funciona el agente, paso a paso
Etapa 1 — Ingesta del documento (ingest.py)
Carga: PyPDFLoader (LangChain) lee el PDF página por página.
Fragmentación (chunking): RecursiveCharacterTextSplitter divide el texto en fragmentos de ~1000 caracteres, con un solape (overlap) de 150 caracteres entre fragmentos consecutivos, para no perder contexto en los cortes.
Embeddings: cada fragmento se convierte en un vector numérico (1536 dimensiones) usando el modelo gemini-embedding-001 de Google — estos vectores capturan el significado del texto, no solo las palabras.
Indexación: los vectores se guardan en un índice FAISS (Facebook AI Similarity Search), una base de datos vectorial que corre 100% local y permite búsquedas de similitud casi instantáneas.
Etapa 2 — Conversación (agent.py / app.py)
El usuario escribe una pregunta en lenguaje natural.
La pregunta se convierte también en un vector de embeddings.
El retriever compara ese vector contra el índice FAISS y recupera los 4 fragmentos más similares (los que más probablemente contienen la respuesta).
Esos fragmentos se insertan como contexto en un prompt junto con la pregunta original, y se envían al modelo de chat gemini-flash-latest.
El modelo genera una respuesta en lenguaje natural, con una instrucción explícita de no inventar información que no esté en el contexto recibido — si no encuentra la respuesta en los fragmentos, lo dice claramente en vez de alucinar una respuesta falsa.
La respuesta se muestra junto con la(s) página(s) del PDF de donde se extrajo, para que el usuario pueda verificarla.

💡 Este diseño (recuperar antes de generar, y prohibir inventar datos) es lo que distingue a un agente RAG confiable de un chatbot genérico: las respuestas están ancladas a una fuente verificable.

🛠️ Tecnologías
Componente	Tecnología
Lenguaje	Python 3.12
Orquestación del agente	LangChain
Lectura de PDF	PyPDF
Base vectorial	FAISS (local, en disco)
Modelo de embeddings	Gemini (gemini-embedding-001)
Modelo de lenguaje (LLM)	Gemini (gemini-flash-latest)
Interfaz web	Streamlit
Infraestructura / despliegue	Oracle Cloud Infrastructure (OCI Compute)
📁 Estructura del proyecto
alura-agente/
├── data/                  # Documentos PDF de entrada
├── vectorstore/           # Índice FAISS generado (no se sube a git)
├── ingest.py              # Etapa 1: procesamiento del documento
├── agent.py               # Etapa 2: agente conversacional (consola)
├── app.py                 # Etapa 2: interfaz web (Streamlit)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
🚀 Cómo ejecutarlo localmente
Clonar el repositorio y crear un entorno virtual
bash
   git clone <url-de-tu-repo>
   cd alura-agente
   python -m venv venv
   source venv/bin/activate   # En Windows: venv\Scripts\Activate.ps1
Instalar dependencias
bash
   pip install -r requirements.txt
Configurar la clave de API
bash
   cp .env.example .env

Edita .env y coloca tu clave gratuita de Google AI Studio.

Colocar tu documento Copia tu PDF dentro de la carpeta data/, por ejemplo: data/documentacion_tecnica_nimbuspay.pdf
Procesar el documento (Etapa 1)
bash
   python ingest.py data/documentacion_tecnica_nimbuspay.pdf
Conversar con el agente por consola (Etapa 2)
bash
   python agent.py
Interfaz web (Streamlit)

Además del chat por consola, el proyecto incluye una interfaz web con Streamlit, con tema oscuro/claro y gestión de documentos desde el navegador:

bash
streamlit run app.py

Se abre automáticamente en http://localhost:8501. Desde ahí puedes:

Subir un nuevo PDF o elegir uno ya existente en data/
Procesarlo con un clic (equivalente a ingest.py)
Conversar con el agente en una interfaz de chat, viendo la página fuente de cada respuesta
Cambiar entre modo oscuro y modo claro con el botón ☀️/🌙
 
Ejemplos de preguntas y respuestas

Pregunta: ¿Qué lenguaje se usa en el back-end de NimbusPay? Agente: El lenguaje principal del back-end es Python, con el framework FastAPI. Para el motor de conciliación bancaria se usa Go por rendimiento y concurrencia. 📎 Fuente: página 1

Pregunta: ¿En qué nube corre la infraestructura y qué orquestador usan? Agente: Toda la infraestructura corre sobre Oracle Cloud Infrastructure (OCI), con los microservicios desplegados como contenedores Docker orquestados con Kubernetes (OKE). 📎 Fuente: página 2

Pregunta: ¿El documento menciona si usan Django? Agente: No encontré esa información en los documentos proporcionados.

(Reemplaza estos ejemplos por preguntas y respuestas reales de tu propio documento antes de la entrega final).

 Despliegue

Nota sobre la infraestructura elegida: el enunciado del challenge sugiere Oracle Cloud Infrastructure (OCI) como opción de despliegue, pero aclara explícitamente que es una sugerencia y no una obligación. Se intentó el despliegue en una instancia VM.Standard.A1.Flex de OCI (documentado más abajo), pero la región de Bogotá presentó falta de capacidad disponible de forma persistente para ese shape (error "Out of capacity", una limitación de infraestructura de Oracle, no del proyecto). Por esa razón, el despliegue final se realizó en Streamlit Community Cloud, una plataforma gratuita más adecuada para una app Streamlit como esta.

 Aplicación desplegada

URL pública: pendiente — se agrega en cuanto finalice el despliegue

Pasos del despliegue en Streamlit Community Cloud
Subir el repositorio a GitHub (con .env excluido vía .gitignore).
Ingresar a share.streamlit.io con la cuenta de GitHub.
Crear una nueva app, seleccionando este repositorio, la rama main y el archivo app.py como punto de entrada.
Configurar la clave de API como Secret (GOOGLE_API_KEY) desde el panel de configuración de la app, en lugar de subir el archivo .env.
Desplegar. Streamlit instala automáticamente las dependencias de requirements.txt y publica la app en una URL pública.
Intento de despliegue en OCI (referencia)

Se documentó y probó la siguiente configuración en OCI, por si se retoma en el futuro cuando haya capacidad disponible:

Parámetro	Valor
Shape	VM.Standard.A1.Flex (Always Free-eligible)
Recursos	1 OCPU / 6 GB RAM
Boot volume	50 GB
Sistema operativo	Oracle Linux 9
Región	South America Central (Bogotá)

Pasos generales: crear la instancia con SSH keys → conectarse por SSH → instalar Python y git → clonar el repositorio → configurar .env → correr ingest.py → abrir el puerto 8501 en las reglas de seguridad → levantar streamlit run app.py --server.port 8501 --server.address 0.0.0.0.

Limitaciones conocidas
Las respuestas dependen enteramente del contenido del PDF indexado; el agente está instruido para no inventar información que no esté en el documento.
El índice FAISS se genera localmente en disco; si se despliega en más de una instancia, cada una necesita su propia copia del índice o un proceso de sincronización.
La capacidad gratuita de instancias A1.Flex en algunas regiones de OCI (incluida Bogotá) puede estar limitada por alta demanda, lo que puede retrasar la creación de la instancia (error "Out of capacity"); no es un problema de configuración del proyecto.