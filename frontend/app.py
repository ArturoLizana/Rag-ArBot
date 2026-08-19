import streamlit as st
import requests

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

API_BASE_URL = "https://rag-backend-wp9f.onrender.com"

# --- Style CSS personnalisé ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    
    /* Alignment du message utilisateur à droite */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
        flex-direction: row-reverse;
        text-align: right;
    }
    
    /* Conteneur de message utilisateur ajusté pour le côté droit */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) div[data-testid="stChatMessageContent"] {
        margin-left: auto;
        margin-right: 0;
        background-color: #1E293B;
        border-radius: 12px 12px 0px 12px;
        padding: 10px 14px;
        display: inline-block;
    }

    /* Conteneur de message assistant ajusté pour le côté gauche */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) div[data-testid="stChatMessageContent"] {
        background-color: #0F172A;
        border-radius: 12px 12px 12px 0px;
        padding: 10px 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Header de l'application
st.markdown('<div class="main-title">🤖 Assistant RAG d\'Entreprise</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interrogez vos documents (PDF, Word) et vos API REST en temps réel.</div>', unsafe_allow_html=True)

# --- Barre latérale : Gestion des sources de données ---
with st.sidebar:
    st.header("📥 Gestions des Connaissances")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📄 Fichiers (PDF/Word)", "🌐 API REST JSON"])
    
    # --- Onglet 1 : Ingestion de Fichiers ---
    with tab1:
        st.subheader("Téléverser un document")
        uploaded_file = st.file_uploader("Choisissez un fichier", type=["pdf", "docx"])
        
        if uploaded_file is not None:
            if st.button("Indexer le fichier", use_container_width=True):
                with st.spinner("Traitement et indexation vectorielle en cours..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_BASE_URL}/ingest/file", files=files)
                        
                        if response.status_code == 200:
                            st.success(f"✅ {uploaded_file.name} indexé avec succès !")
                        else:
                            st.error(f"❌ Erreur : {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"❌ Impossible de contacter le serveur backend : {e}")

    # --- Onglet 2 : Ingestion via API REST ---
    with tab2:
        st.subheader("Ingérer une API REST")
        api_url = st.text_input("URL de l'API REST", placeholder="https://api.example.com/data")
        text_key = st.text_input("Clé JSON cible (Optionnel)", placeholder="ex: body, description...")
        
        if st.button("Indexer l'API", use_container_width=True):
            if not api_url:
                st.warning("Veuillez saisir une URL valide.")
            else:
                with st.spinner("Récupération des données API et indexation..."):
                    try:
                        payload = {"url": api_url, "text_key": text_key if text_key else None}
                        response = requests.post(f"{API_BASE_URL}/ingest/api", json=payload)
                        
                        if response.status_code == 200:
                            st.success("✅ Données de l'API indexées avec succès !")
                        else:
                            st.error(f"❌ Erreur : {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"❌ Erreur de connexion au backend : {e}")

    st.markdown("---")
    st.markdown("⚙️ **Paramètres de recherche**")
    top_k = st.slider("Nombre de contextes extraits (Top K)", min_value=1, max_value=10, value=3)

# --- Zone Principale : Interface de Chat ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources documentaires utilisées"):
                for idx, src in enumerate(message["sources"], 1):
                    src_name = src.get("source", "Source inconnue")
                    page = f" (Page {src['page'] + 1})" if src.get("page") is not None else ""
                    st.markdown(f"**{idx}.** `{src_name}`{page}")

# Saisie utilisateur
if prompt := st.chat_input("Posez votre question sur les documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la base de connaissances et génération..."):
            try:
                payload = {"question": prompt, "top_k": top_k}
                response = requests.post(f"{API_BASE_URL}/query", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Aucune réponse générée.")
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("📚 Sources documentaires utilisées"):
                            for idx, src in enumerate(sources, 1):
                                src_name = src.get("source", "Source inconnue")
                                page = f" (Page {src['page'] + 1})" if src.get("page") is not None else ""
                                st.markdown(f"**{idx}.** `{src_name}`{page}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    err_msg = response.json().get("detail", "Erreur serveur.")
                    st.error(f"❌ {err_msg}")

            except Exception as e:
                st.error(f"❌ Impossible de se connecter à l'API Backend FastAPI : {e}")