# MistralChat.py (version HYBRIDE)
import streamlit as st
import os
import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
import time

# --- Router & SQL Tool ---

from scripts.sql_tool import NBADataTool, execute_sql_query_standalone
from scripts.router import QuestionRouter, HybridQueryExecutor

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME
    )
    from utils.vector_store import VectorStoreManager
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()
# --- Logfire ---

import logfire
logfire.configure(service_name="nba-rag")

# --- Configuration du Logging ---
# Note: Streamlit peut avoir sa propre gestion de logs. Configurer ici est une bonne pratique.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration de l'API Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()

try:
    client = MistralClient(api_key=api_key)
    logging.info("Client Mistral initialisé.")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    logging.exception("Erreur initialisation client Mistral")
    st.stop()

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource # Garde le manager chargé en mémoire pour la session
def get_vector_store_manager():
    with logfire.span("load_vector_store"):
        logging.info("Tentative de chargement du VectorStoreManager...")
        try:
            manager = VectorStoreManager()
            # Vérifie si l'index a bien été chargé par le constructeur
            if manager.index is None or not manager.document_chunks:
                st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
                st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
                logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
                return None # Retourne None si échec
            #Success Log 
            logfire.info("Vector store loaded", vectors=manager.index.ntotal if manager.index else 0 )
            logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
            return manager
        except FileNotFoundError:
            st.error("Fichiers d'index ou de chunks non trouvés.")
            st.warning("Veuillez exécuter 'python indexer.py' pour créer la base de connaissances.")
            logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
            return None
        except Exception as e:
            logfire.exception("Vector store loading failed")
            st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
            logging.exception("Erreur chargement VectorStoreManager")
            return None
        
vector_store_manager = get_vector_store_manager()

# === Initialisation du SQL Tool, Router, et Exécuteur Hybride ===

sql_tool = NBADataTool(client)
router = QuestionRouter(client)
hybrid_executor = HybridQueryExecutor(
    mistral_client=client,
    vector_store_manager=vector_store_manager,
    sql_tool=sql_tool
)


# --- Prompt Système pour RAG ---
# Adaptez ce prompt selon vos besoins
SYSTEM_PROMPT = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{{context_str}}
---

QUESTION DU FAN:
{{question}}

RÉPONSE DE L'ANALYSTE NBA:"""


# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    # Message d'accueil initial
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]

# --- Fonctions ---

def generer_reponse(prompt_messages: list[ChatMessage]) -> str:
    """
    Envoie le prompt (qui inclut maintenant le contexte) à l'API Mistral.
    """
    with logfire.span("llm_call") as span:
        span.set_attribute("model", model)
        span.set_attribute("messages_count", len(prompt_messages))

        if not prompt_messages:
            logging.warning("Tentative de génération de réponse avec un prompt vide.")
            return "Je ne peux pas traiter une demande vide."
        
        try:
            logging.info(f"Appel à l'API Mistral modèle '{model}' avec {len(prompt_messages)} message(s).")
            # Log le contenu du prompt (peut être long) - commenter si trop verbeux
            # logging.debug(f"Prompt envoyé à l'API: {prompt_messages}")

            response = client.chat(
                model=model,
                messages=prompt_messages,
                temperature=0.1, # Température basse pour des réponses factuelles basées sur le contexte
                # top_p=0.9,
                )
            if response.choices and len(response.choices) > 0:
                logging.info("Réponse reçue de l'API Mistral.")
                logfire.info("LLM response received", answer_length=len(response.choices))
                return response.choices[0].message.content
            else:
                logging.warning("L'API n'a pas retourné de choix valide.")
                return "Désolé, je n'ai pas pu générer de réponse valide pour le moment."
        except Exception as e:
            logfire.exception("LLM call failed")
            st.error(f"Erreur lors de l'appel à l'API Mistral: {e}")
            logging.exception("Erreur API Mistral pendant client.chat")
            return "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."

# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    startq_timing = time.time()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

        # === ROUTING INTELLIGENT  ===
    start_routing = time.time()
    decision = router.route_question(prompt)
    routing_duration= time.time() - start_routing
    logfire.info("router_latency", seconds=routing_duration)

    # Log
    logging.info(
        f"Routing decision = {decision.strategy} "
        f"(conf={decision.confidence:.2f}) | reasoning = {decision.reasoning}"
    )

    # --- SQL-ONLY ---

    if decision.strategy == "sql_only":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("Analyse SQL en cours...")
            start_sql = time.time()
            sql_response = sql_tool.run(prompt)
            sql_latency = time.time() - start_sql
            logfire.info("sql_latency", seconds=sql_latency)

            cleaned_response = "\n".join(
                line for line in sql_response.split("\n")
                if not line.strip().startswith("Requête SQL exécutée")
                and not line.strip().startswith("SELECT")
                )

            st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
            if "Données:" in cleaned_response:
                header, data = cleaned_response.split("Données:", 1)
                message_placeholder.write(header)
                st.code(data.strip())
            else:
                message_placeholder.write(cleaned_response)
        total_duration_wsql = time.time() - startq_timing
        logfire.info("mistral_response_timing", seconds=total_duration_wsql)
        st.stop()

    # --- RAG-ONLY ---
    if decision.strategy == "rag_only":
        # On laisse continuer vers le pipeline RAG actuel (ne rien changer)
        pass

    # --- HYBRID (SQL + RAG + Synthèse) ---
    if decision.strategy == "hybrid":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.text("Analyse hybride (SQL + RAG) en cours...")
            start_hybride = time.time()
            hybrid_response = hybrid_executor.execute(
                question=prompt,
                decision=decision,
                search_k=SEARCH_K
            )
            hybrid_latency = time.time() - start_hybride
            logfire.info("hybrid_latency", seconds=hybrid_latency)

            st.session_state.messages.append({"role": "assistant", "content": hybrid_response})
            message_placeholder.write(hybrid_response)
        total_duration_whybrid = time.time() - startq_timing
        logfire.info("mistral_response_timing", seconds=total_duration_whybrid)
        st.stop()

    with logfire.span("full_rag_pipeline") as span:

        # Log de la question utilisateur
        span.set_attribute("user_question", prompt)

    # === Début de la logique RAG ===

    # 2. Vérifier si le Vector Store est disponible
        if vector_store_manager is None:
            span.set_attribute("error", "vector_store_not_available")
            st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
            logging.error("VectorStoreManager non disponible pour la recherche.")
            # On arrête ici car on ne peut pas faire de RAG
            st.stop()

    # 3. Rechercher le contexte dans le Vector Store
        try:
            logging.info(f"Recherche de contexte pour la question: '{prompt}' avec k={SEARCH_K}")
            start_rag = time.time()
            search_results = vector_store_manager.search(prompt, k=SEARCH_K)
            rag_latency = time.time() - start_rag
            logfire.info("rag_latency", seconds=rag_latency)
            span.set_attribute("chunks_found", len(search_results))
            logging.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
        except Exception as e:
            span.set_attribute("retrieval_error", str(e))
            st.error(f"Une erreur est survenue lors de la recherche d'informations pertinentes: {e}")
            logging.exception(f"Erreur pendant vector_store_manager.search pour la query: {prompt}")
            search_results = [] # On continue sans contexte si la recherche échoue

    # 4. Formater le contexte pour le prompt LLM
        context_str = "\n\n---\n\n".join([
            f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
            for res in search_results
            ])
        span.set_attribute("context_length", len(context_str))

        if not search_results:
            context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
            logging.warning(f"Aucun contexte trouvé pour la query: {prompt}")

    # 5. Construire le prompt final pour l'API Mistral en utilisant le System Prompt RAG
        final_prompt_for_llm = SYSTEM_PROMPT.format(context_str=context_str, question=prompt)
        logfire.info("Prompt built", context_length=len(context_str), chunks_used=len(search_results))

    # Créer la liste de messages pour l'API (juste le prompt système/utilisateur combiné)
        messages_for_api = [
        # On pourrait séparer system et user, mais Mistral gère bien un long message user structuré
            ChatMessage(role="user", content=final_prompt_for_llm)
    ]

    # === Fin de la logique RAG ===


    # 6. Afficher indicateur + Générer la réponse de l'assistant via LLM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Indicateur simple

        # Génération de la réponse de l'assistant en utilisant le prompt augmenté
        response_content = generer_reponse(messages_for_api)
        
        # Logfire : taille de la réponse
        span.set_attribute("answer_length", len(response_content))
        span.set_attribute("status", "success")

        # Affichage de la réponse complète
        message_placeholder.write(response_content)
        
    response_duration = time.time() - startq_timing
    logfire.info("mistral_response_timing", seconds=response_duration)

    # 7. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI & Faiss | Data-driven NBA Insights")