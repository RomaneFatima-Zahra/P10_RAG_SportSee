# router.py
"""
Router intelligent pour gérer les questions mixtes (qualitatives + quantitatives).
Détermine quelle source utiliser : Vector Store (RAG), SQL DB, ou les deux.
"""

import logging
from typing import Dict, List, Literal
from dataclasses import dataclass
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from utils.config import DB_CONFIG, MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Décision de routing pour une question"""
    strategy: Literal["rag_only", "sql_only", "hybrid"]
    confidence: float
    reasoning: str
    sql_subquestion: str = None
    rag_subquestion: str = None


class QuestionRouter:
    """
    Router intelligent qui analyse les questions et détermine la stratégie optimale.
    Gère 3 cas :
    1. RAG_ONLY : questions qualitatives (opinions, contexte, histoire)
    2. SQL_ONLY : questions purement quantitatives (stats, chiffres)
    3. HYBRID : questions mixtes nécessitant les deux sources
    """
    
    ROUTING_PROMPT = """Tu es un expert en analyse de questions NBA. Ta tâche est de déterminer quelle source de données utiliser.

SOURCES DISPONIBLES:
1. **RAG (Vector Store)** : Discussions Reddit, opinions de fans, contexte historique, analyses qualitatives
2. **SQL (Database)** : Statistiques chiffrées, performances de joueurs/équipes, données quantitatives
3. **HYBRID** : Combinaison des deux pour questions mixtes

EXEMPLES DE CLASSIFICATION:

Question: "Qui est considéré comme le GOAT par les fans ?"
Stratégie: rag_only
Raisonnement: Question d'opinion, pas de statistiques précises

Question: "Combien de points LeBron a marqué cette saison ?"
Stratégie: sql_only
Raisonnement: Question purement statistique

Question: "Pourquoi les Lakers sont-ils meilleurs cette année ? Compare leurs stats à l'an dernier"
Stratégie: hybrid
SQL: "Statistiques des Lakers 2024 vs 2023"
RAG: "Analyse qualitative de la performance des Lakers"
Raisonnement: Nécessite stats + contexte/analyses

Question: "Les fans pensent-ils que Curry peut encore gagner un titre ?"
Stratégie: rag_only
Raisonnement: Opinion des fans

Question: "Quelle est la moyenne de points des Warriors ?"
Stratégie: sql_only
Raisonnement: Statistique précise

Question: "LeBron est-il toujours au top ? Montre-moi ses stats et ce que les gens en disent"
Stratégie: hybrid
SQL: "Statistiques actuelles de LeBron James"
RAG: "Opinions et analyses sur LeBron James"
Raisonnement: Combine données chiffrées et opinions

INSTRUCTIONS:
Analyse la question et réponds UNIQUEMENT avec un objet JSON valide.
NE PAS AJOUTER de texte avant ou après le JSON.
NE PAS utiliser de markdown code blocks.

Format exact attendu:
{{
  "strategy": "rag_only",
  "confidence": 0.9,
  "reasoning": "explication courte",
  "sql_subquestion": null,
  "rag_subquestion": null
}}

Valeurs possibles pour "strategy": "rag_only", "sql_only", ou "hybrid"
Valeurs pour "confidence": nombre entre 0.0 et 1.0
Si strategy n'est pas "hybrid", mets sql_subquestion et rag_subquestion à null

QUESTION À ANALYSER:
{question}

Réponds UNIQUEMENT avec le JSON (sans texte additionnel):"""

    def __init__(self, mistral_client: MistralClient):
        """
        Initialise le router.
        
        Args:
            mistral_client: Client Mistral pour l'analyse de questions
        """
        self.mistral_client = mistral_client
        logger.info("QuestionRouter initialisé")
    
    def route_question(self, question: str) -> RoutingDecision:
        """
        Analyse une question et détermine la stratégie de routing.
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            RoutingDecision avec la stratégie et les sous-questions si nécessaire
        """
        logger.info(f"Routing de la question: '{question}'")
        
        try:
            # Appel au LLM pour analyser la question
            prompt = self.ROUTING_PROMPT.format(question=question)
            messages = [ChatMessage(role="user", content=prompt)]
            
            response = self.mistral_client.chat(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parser la réponse JSON en nettoyant le contenu
            import json
            import re
            
            content = response.choices[0].message.content.strip()
            logger.debug(f"Réponse brute du LLM: {content[:200]}...")
            
            # Extraire le JSON entre les accolades
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                raise ValueError("Aucun JSON trouvé dans la réponse")
            
            json_str = json_match.group(0)
            
            # Nettoyer les éventuels markdown code blocks
            json_str = json_str.replace('```json', '').replace('```', '')
            
            result = json.loads(json_str)
            
            decision = RoutingDecision(
                strategy=result["strategy"],
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                sql_subquestion=result.get("sql_subquestion"),
                rag_subquestion=result.get("rag_subquestion")
            )
            
            logger.info(f"Stratégie: {decision.strategy} (confiance: {decision.confidence:.2f})")
            logger.info(f"Raisonnement: {decision.reasoning}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Erreur lors du routing: {e}")
            logger.debug(f"Détails de l'erreur: {str(e)}")
            # Fallback : utiliser une heuristique simple
            return self._fallback_routing(question)
    
    def _fallback_routing(self, question: str) -> RoutingDecision:
        """
        Méthode de secours basée sur des mots-clés si le LLM échoue.
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            RoutingDecision basée sur l'analyse par mots-clés
        """
        logger.warning("Utilisation du routing de secours (mots-clés)")
        
        question_lower = question.lower()
        
        # Mots-clés SQL
        sql_keywords = [
            'combien', 'nombre', 'moyenne', 'total', 'statistiques', 'stats',
            'points', 'rebonds', 'passes', 'pourcentage', 'top', 'classement',
            'compare', 'différence', 'meilleur scoreur', 'efficacité'
        ]
        
        # Mots-clés RAG
        rag_keywords = [
            'pourquoi', 'comment', 'opinion', 'pense', 'considère', 'fans',
            'débat', 'controversé', 'histoire', 'contexte', 'analyse',
            'goat', 'légende', 'discutent', 'disent', 'sentiment'
        ]
        
        # Mots-clés Hybrid
        hybrid_keywords = [
            'et', 'mais aussi', 'ainsi que', 'compare', 'montre-moi',
            'explique avec', 'justifie', 'prouve'
        ]
        
        sql_score = sum(1 for kw in sql_keywords if kw in question_lower)
        rag_score = sum(1 for kw in rag_keywords if kw in question_lower)
        hybrid_score = sum(1 for kw in hybrid_keywords if kw in question_lower)
        
        # Décision
        if hybrid_score > 0 and sql_score > 0 and rag_score > 0:
            strategy = "hybrid"
            confidence = 0.6
            reasoning = "Question mixte détectée par mots-clés"
        elif sql_score > rag_score:
            strategy = "sql_only"
            confidence = 0.7
            reasoning = "Question quantitative détectée par mots-clés"
        else:
            strategy = "rag_only"
            confidence = 0.7
            reasoning = "Question qualitative détectée par mots-clés"
        
        return RoutingDecision(
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning
        )


class HybridQueryExecutor:
    """
    Exécute les requêtes hybrides en combinant RAG et SQL.
    """
    
    SYNTHESIS_PROMPT = """Tu es un analyste NBA expert. Tu dois synthétiser des informations provenant de deux sources:

1. **DONNÉES STATISTIQUES (Base SQL)**:
{sql_results}

2. **ANALYSES ET OPINIONS (Discussions Reddit)**:
{rag_results}

QUESTION ORIGINALE: {question}

INSTRUCTIONS:
- Commence par les données chiffrées (SQL) pour établir les faits
- Enrichis avec le contexte et les opinions (RAG)
- Crée une réponse cohérente qui combine les deux
- Cite les sources quand pertinent
- Reste factuel et objectif

RÉPONSE SYNTHÉTISÉE:"""

    def __init__(self, mistral_client: MistralClient, vector_store_manager, sql_tool):
        """
        Initialise l'exécuteur hybride.
        
        Args:
            mistral_client: Client Mistral
            vector_store_manager: Manager du vector store (RAG)
            sql_tool: Tool SQL pour les requêtes quantitatives
        """
        self.mistral_client = mistral_client
        self.vector_store = vector_store_manager
        self.sql_tool = sql_tool
        logger.info("HybridQueryExecutor initialisé")
    
    def execute(self, question: str, decision: RoutingDecision, search_k: int = 3) -> str:
        """
        Exécute une requête selon la stratégie de routing.
        
        Args:
            question: Question originale
            decision: Décision de routing
            search_k: Nombre de chunks RAG à récupérer
            
        Returns:
            Réponse finale synthétisée
        """
        logger.info(f"Exécution de la stratégie: {decision.strategy}")
        
        if decision.strategy == "rag_only":
            return self._execute_rag_only(question, search_k)
        
        elif decision.strategy == "sql_only":
            return self._execute_sql_only(question)
        
        elif decision.strategy == "hybrid":
            return self._execute_hybrid(question, decision, search_k)
        
        else:
            logger.error(f"Stratégie inconnue: {decision.strategy}")
            return "Erreur: stratégie de routing inconnue"
    
    def _execute_rag_only(self, question: str, k: int) -> str:
        """Exécute une requête RAG uniquement"""
        logger.info("Exécution RAG uniquement")
        
        try:
            search_results = self.vector_store.search(question, k=k)
            
            if not search_results:
                return "❌ Aucune information pertinente trouvée dans les discussions."
            
            context = "\n\n".join([
                f"Source: {r['metadata'].get('source', 'Inconnue')} (Score: {r['score']:.1f}%)\n{r['text']}"
                for r in search_results
            ])
            
            return f"📚 CONTEXTE DES DISCUSSIONS:\n\n{context}"
            
        except Exception as e:
            logger.error(f"Erreur RAG: {e}")
            return f"❌ Erreur lors de la recherche RAG: {e}"
    
    def _execute_sql_only(self, question: str) -> str:
        """Exécute une requête SQL uniquement"""
        logger.info("Exécution SQL uniquement")
        
        try:
            return self.sql_tool.run(question)
        except Exception as e:
            logger.error(f"Erreur SQL: {e}")
            return f"❌ Erreur lors de l'exécution SQL: {e}"
    
    def _execute_hybrid(self, question: str, decision: RoutingDecision, k: int) -> str:
        """Exécute une requête hybride (RAG + SQL)"""
        logger.info("Exécution HYBRID (RAG + SQL)")
        
        try:
            # 1. Requête SQL
            sql_subquestion = decision.sql_subquestion or question
            logger.info(f"Sous-question SQL: {sql_subquestion}")
            sql_results = self.sql_tool.run(sql_subquestion)
            
            # 2. Requête RAG
            rag_subquestion = decision.rag_subquestion or question
            logger.info(f"Sous-question RAG: {rag_subquestion}")
            search_results = self.vector_store.search(rag_subquestion, k=k)
            
            if not search_results:
                rag_results = "Aucune discussion pertinente trouvée."
            else:
                rag_results = "\n\n".join([
                    f"- {r['text'][:200]}... (Source: {r['metadata'].get('source', 'N/A')})"
                    for r in search_results
                ])
            
            # 3. Synthèse par le LLM
            synthesis_prompt = self.SYNTHESIS_PROMPT.format(
                sql_results=sql_results,
                rag_results=rag_results,
                question=question
            )
            
            messages = [ChatMessage(role="user", content=synthesis_prompt)]
            
            response = self.mistral_client.chat(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )
            
            synthesized_answer = response.choices[0].message.content
            
            return f"🔄 ANALYSE COMPLÈTE (Stats + Contexte):\n\n{synthesized_answer}"
            
        except Exception as e:
            logger.error(f"Erreur HYBRID: {e}")
            return f"❌ Erreur lors de l'exécution hybride: {e}"


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("🧪 Tests du Router NBA")
    print("=" * 80)
    
    # Créer un mock client pour les tests
    from utils.config import MISTRAL_API_KEY
    client = MistralClient(api_key=MISTRAL_API_KEY)
    
    router = QuestionRouter(client)
    
    # Questions de test
    test_questions = [
        "Qui est le GOAT selon les fans ?",
        "Combien de points LeBron a marqué cette saison ?",
        "Pourquoi les Lakers sont meilleurs ? Montre-moi leurs stats",
        "Compare les performances de Curry et les opinions sur lui",
        "Top 5 des scoreurs",
        "Qu'est-ce que les gens pensent du trade des Nets ?"
    ]
    
    print("\n📋 Test de routing:\n")
    for q in test_questions:
        print(f"\nQuestion: {q}")
        decision = router.route_question(q)
        print(f"  → Stratégie: {decision.strategy}")
        print(f"  → Confiance: {decision.confidence:.2f}")
        print(f"  → Raisonnement: {decision.reasoning}")
        if decision.sql_subquestion:
            print(f"  → SQL: {decision.sql_subquestion}")
        if decision.rag_subquestion:
            print(f"  → RAG: {decision.rag_subquestion}")
    
    print("\n" + "=" * 80)
    print("✅ Tests terminés")