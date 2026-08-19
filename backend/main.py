import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# 1. Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Import du module d'ingestion local
from backend.ingest import ingest_source, CHROMA_DB_DIR

app = FastAPI(
    title="RAG Enterprise API",
    description="API REST pour le traitement de documents et le RAG (PDF, Word, API JSON)",
    version="1.0.0"
)

# ACTIVATION DU CORS (Obligatoire pour l'interface Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement des embeddings via l'API HF (ultra-léger, sans PyTorch)
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_token,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Récupérer la clé d'API depuis .env
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("La variable GROQ_API_KEY est introuvable ou vide dans le fichier .env")

# Injection dans les variables d'environnement système pour LangChain/Groq
os.environ["GROQ_API_KEY"] = groq_api_key

# 3. Initialisation exactement avec le modèle qui fonctionnait chez toi
llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class APIIngestRequest(BaseModel):
    url: str
    text_key: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API Enterprise RAG. Consultez /docs pour la documentation Swagger."}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Endpoint pour ingérer un fichier PDF ou Word téléversé."""
    allowed_extensions = [".pdf", ".docx"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail="Extension non supportée. Seuls les fichiers .pdf et .docx sont acceptés."
        )

    # Sauvegarde temporaire du fichier dans le dossier /data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Lancement du processus d'ingestion
        ingest_source(file_path)
        return {
            "status": "success",
            "message": f"Fichier '{file.filename}' indexé avec succès.",
            "path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'ingestion : {str(e)}")


@app.post("/ingest/api")
async def ingest_api(request: APIIngestRequest):
    """Endpoint pour ingérer le contenu JSON d'une API REST externe."""
    try:
        ingest_source(request.url, is_api=True, api_text_key=request.text_key)
        return {
            "status": "success",
            "message": f"Données de l'API '{request.url}' indexées avec succès."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'ingestion API : {str(e)}")


@app.post("/query")
async def query_rag(request: QueryRequest):
    """Endpoint pour poser une question au RAG."""
    if not os.path.exists(CHROMA_DB_DIR):
        raise HTTPException(
            status_code=400, 
            detail="La base de données vectorielle est vide. Veuillez d'abord ingérer des documents."
        )

    try:
        # 1. Connexion à la Vector DB
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": request.top_k})

        # 2. Recherche des contextes pertinents
        docs = retriever.invoke(request.question)

        if not docs:
            return {
                "answer": "Aucun document pertinent trouvé dans la base de connaissances.",
                "sources": []
            }

        # 3. Assemblage du contexte
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # 4. Prompt d'entreprise avec garde-fous
        prompt = f"""Tu es un assistant virtuel d'entreprise rigoureux.
Réponds à la question en t'appuyant uniquement sur le contexte ci-dessous.
Si le contexte ne contient pas la réponse, réponds strictly : "L'information n'est pas présente dans les documents fournis."

Contexte :
{context_text}

Question : {request.question}
Réponse :"""

        # 5. Génération par le LLM
        response = llm.invoke(prompt)

        # Extraction du texte propre si la réponse est un objet AIMessage
        answer_text = response.content if hasattr(response, "content") else str(response)

        # Extraction des métadonnées des sources
        sources = [
            {
                "source": doc.metadata.get("source", "Inconnue"),
                "page": doc.metadata.get("page", None)
            }
            for doc in docs
        ]

        return {
            "answer": answer_text,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement : {str(e)}")