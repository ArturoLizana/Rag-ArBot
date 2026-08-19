import os
import requests
from dotenv import load_dotenv
from typing import List, Union
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Charger explicitement les variables d'environnement
load_dotenv()

# Dossier par défaut pour persister ChromaDB
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# 2. Récupération sécurisée du token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token
)


def load_pdf(file_path: str) -> List[Document]:
    """Charge un fichier PDF local."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_docx(file_path: str) -> List[Document]:
    """Charge un fichier Word (.docx) local."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier non trouvé : {file_path}")
    loader = Docx2txtLoader(file_path)
    return loader.load()


def load_json_api(api_url: str, text_key: str = None) -> List[Document]:
    """
    Récupère des données JSON depuis une API REST et les transforme en Documents LangChain.
    
    :param api_url: URL de l'API REST
    :param text_key: Clé spécifique du JSON à utiliser pour le texte (optionnel)
    """
    response = requests.get(api_url)
    response.raise_for_status()
    data = response.json()

    documents = []
    
    # Si la réponse de l'API est une liste d'objets JSON
    if isinstance(data, list):
        for index, item in enumerate(data):
            if text_key and text_key in item:
                content = str(item[text_key])
            else:
                content = str(item)
            
            doc = Document(
                page_content=content,
                metadata={"source": api_url, "item_index": index}
            )
            documents.append(doc)
            
    # Si la réponse de l'API est un objet JSON unique
    elif isinstance(data, dict):
        if text_key and text_key in data:
            content = str(data[text_key])
        else:
            content = str(data)
            
        doc = Document(
            page_content=content,
            metadata={"source": api_url}
        )
        documents.append(doc)
        
    return documents


def process_and_index_documents(
    docs: List[Document], 
    db_dir: str = CHROMA_DB_DIR,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Chroma:
    """
    Découpe les documents en morceaux (chunks) et les sauvegarde dans ChromaDB.
    """
    # 1. Découpage intelligent du texte (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"--> Document découpé en {len(chunks)} morceaux (chunks).")

    # 2. Indexation vectorielle dans ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    print(f"--> Indexation terminée et sauvegardée dans : {db_dir}")
    return vectorstore


def ingest_source(source_path_or_url: str, is_api: bool = False, api_text_key: str = None):
    """
    Point d'entrée principal pour l'ingestion automatique de n'importe quelle source.
    """
    print(f"\n[Ingestion] Traitement de : {source_path_or_url}")
    
    if is_api:
        docs = load_json_api(source_path_or_url, text_key=api_text_key)
    elif source_path_or_url.lower().endswith(".pdf"):
        docs = load_pdf(source_path_or_url)
    elif source_path_or_url.lower().endswith(".docx"):
        docs = load_docx(source_path_or_url)
    else:
        raise ValueError("Format non supporté. Fournir un PDF, Word (.docx) ou une URL API.")

    return process_and_index_documents(docs)


if __name__ == "__main__":
    # --- Test local du module dans le terminal ---
    print("Module ingest.py prêt.")
    # Exemple d'utilisation API REST gratuite pour tester :
    # ingest_source("https://jsonplaceholder.typicode.com/posts", is_api=True, api_text_key="body")