# evaluate_ragas.py

"""
Script d'évaluation du système RAG hybride (RAG + SQL) basé sur le framework  RAGAS.
Mesure la pertinence, la fidélité et la cohérence des réponses après intégration du routing.
"""
import os
import platform

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import time
from ragas.run_config import RunConfig

# Imports RAGAS
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness)
from datasets import Dataset

# Imports locaux
Root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(Root_path))

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K, EMBEDDING_MODEL
from utils.vector_store import VectorStoreManager

# LangChain Mistral (UNIQUE STACK)
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.messages import HumanMessage

# Client Mistral natif pour le router
from mistralai.client import MistralClient

# Imports pour le système hybride
from scripts.sql_tool import NBADataTool
from scripts.router import QuestionRouter, HybridQueryExecutor


# --------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------
# Charger les variables d'environnement
# --------------------------------------------------------------------

load_dotenv()

# --------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------

Dataset_path = Root_path / "Ragas_eval" / "rag_dataset.json"
Output_path = Root_path / "Ragas_eval"  / "rag_results.json"

# --------------------------------------------------------------------
# Script 
# --------------------------------------------------------------------

class Evaluator:

    """Classe pour évaluer le système RAG hybride (Vector Store + SQL) avec RAGAS."""
    
    def __init__(self):
        """Initialise l'évaluateur avec le Vector Store, , SQL Tool et Router."""
        # Vector Store Manager (RAG)
        self.vector_store = VectorStoreManager()

        #  LLM LangChain
        self.llm = ChatMistralAI(
            api_key=MISTRAL_API_KEY,
            model=MODEL_NAME,
            temperature=0.1,
            max_retries=3,
            timeout=30

        )

        #  Embeddings LangChain
        self.embeddings = MistralAIEmbeddings(
            api_key=MISTRAL_API_KEY,
            model=EMBEDDING_MODEL,
            max_retries=3,
            timeout=30
            )
        
        #  Client Mistral natif (pour SQL Tool et Router)
        self.mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
        
        # SQL Tool
        self.sql_tool = NBADataTool(self.mistral_client)
        
        # Router
        self.router = QuestionRouter(self.mistral_client)
        
        # Hybrid Executor
        self.hybrid_executor = HybridQueryExecutor(
            mistral_client=self.mistral_client,
            vector_store_manager=self.vector_store,
            sql_tool=self.sql_tool
        )

        # Vérifier que l'index est chargé
        if self.vector_store.index is None or not self.vector_store.document_chunks:
            raise ValueError("Le Vector Store n'est pas initialisé. Exécutez 'python indexer.py' d'abord.")
        
        logging.info(f"RAGEvaluator initialisé avec {self.vector_store.index.ntotal} vecteurs.")
        logging.info(f" HybridRAGEvaluator initialisé avec {self.vector_store.index.ntotal} vecteurs.")
        logging.info(f" SQL Tool et Router intégrés")

    def generate_answer_with_routing(self, question: str) -> tuple[str, str, List[str]]:
        """
        Génère une réponse en utilisant le système de routing intelligent.
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            tuple: (answer, strategy_used, contexts)
        """
        try:
            # 1. Router la question
            decision = self.router.route_question(question)
            logging.info(f"🔀 Stratégie choisie: {decision.strategy} (confiance: {decision.confidence:.2f})")
            
            # 2. Exécuter selon la stratégie
            if decision.strategy == "rag_only":
                # RAG uniquement
                contexts, _ = self.retrieve_context(question)
                context_str = "\n\n".join(contexts)
                answer = self.generate_answer(question, context_str)
                strategy = "RAG_ONLY"
                
            elif decision.strategy == "sql_only":
                # SQL uniquement
                answer = self.sql_tool.run(question)
                contexts = [answer]  # Le résultat SQL devient le contexte
                strategy = "SQL_ONLY"
                
            elif decision.strategy == "hybrid":
                # Hybride (RAG + SQL)
                answer = self.hybrid_executor.execute(question, decision, search_k=SEARCH_K)
                
                # Récupérer les contextes pour RAGAS (combinaison RAG + SQL)
                rag_contexts, _ = self.retrieve_context(question, k=SEARCH_K)
                sql_result = self.sql_tool.run(decision.sql_subquestion or question)
                contexts = rag_contexts + [f"SQL Result: {sql_result}"]
                strategy = "HYBRID"
            
            else:
                # Fallback
                contexts, _ = self.retrieve_context(question)
                context_str = "\n\n".join(contexts)
                answer = self.generate_answer(question, context_str)
                strategy = "FALLBACK_RAG"
            
            return answer, strategy, contexts
            
        except Exception as e:
            logging.error(f"Erreur lors de la génération avec routing: {e}")
            # Fallback vers RAG classique
            contexts, _ = self.retrieve_context(question)
            context_str = "\n\n".join(contexts)
            answer = self.generate_answer(question, context_str)
            return answer, "ERROR_FALLBACK", contexts
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Génère une réponse à partir de la question et du contexte (méthode RAG classique).
        """
        system_prompt = f"""Tu es 'NBA Analyst AI', un assistant expert sur la ligue NBA.
Tu dois répondre UNIQUEMENT avec le contexte fourni.

CONTEXTE:
{context}

QUESTION:
{question}

RÉPONSE:"""
        
        try:
            messages = [HumanMessage(content=system_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logging.error(f"Erreur lors de la génération de réponse: {e}")
            return ""
    
    def retrieve_context(self, question: str, k: int = SEARCH_K) -> tuple[List[str], List[str]]:
        """Récupère le contexte pertinent pour une question. (RAG classique)"""
        try:
            search_results = self.vector_store.search(question, k=k)
            
            contexts = [res['text'] for res in search_results]
            sources = [res['metadata'].get('source', 'Inconnue') for res in search_results]
            
            logging.info(f"Récupéré {len(contexts)} chunks pour la question.")
            return contexts, sources
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du contexte: {e}")
            return [], []

    # --------------------------------------------------------------------
    # Chargement des questions depuis JSON
    # --------------------------------------------------------------------
    def load_test_questions_from_json(self, json_path: Path = Dataset_path) -> List[Dict]:
        """
        Charge les questions depuis un fichier JSON.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Le fichier JSON doit contenir une liste d'objets.")

            for item in data:
                if "question" not in item:
                    raise ValueError("Chaque entrée doit contenir une clé 'question'.")

            logging.info(f"{len(data)} questions chargées depuis {json_path}")
            return data

        except Exception as e:
            logging.error(f"Erreur lors du chargement des questions JSON: {e}")
            raise

    # --------------------------------------------------------------------
    # evaluation question
    # --------------------------------------------------------------------
    def evaluate_single_question(
        self,
        question: str,
        ground_truth: Optional[str] = None
    ) -> Dict:
        """Évalue une question unique avec le système de routing."""
    
        # Générer la réponse avec le système de routing
        answer, strategy, contexts = self.generate_answer_with_routing(question)
        
        return {
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth if ground_truth else "Non fourni",
            "strategy": strategy
        }

    # --------------------------------------------------------------------
    # Evaluation Complète
    # --------------------------------------------------------------------
    
    def run_evaluation(
        self,
        test_questions: Optional[List[Dict]] = None
    ) -> Dict:
        """Exécute l'évaluation complète avec le système hybride."""

        if test_questions is None:
            test_questions = self.load_test_questions_from_json(Dataset_path)

        logging.info(f"Début de l'évaluation HYBRIDE avec {len(test_questions)} questions...")
        eval_start_time = time.time()  # <-- début du timer global
        
        eval_data = {
            "question": [],
            "contexts": [],
            "answer": [],
            "ground_truth": [],
            "id": [],
            "language": [],
            "duration": [],
            "strategy": []  # ✅ Ajout du suivi de la stratégie
        }
        
        metadata = {
            "categories": [],
            "sources": []
        }
        
        for i, test_q in enumerate(test_questions, 1):
            logging.info(f"Évaluation question {i}/{len(test_questions)}: {test_q['question'][:60]}...")
            q_start = time.time()
             
            # Ajouter un délai pour éviter le rate limiting
            if i > 1:
                time.sleep(2)

            result = self.evaluate_single_question(
                question=test_q["question"],
                ground_truth=test_q.get("ground_truth")
            )
            q_end = time.time()
            q_duration = q_end - q_start
            logging.info(f"⏱️ Temps de réponse: {q_duration:.2f} sec | Stratégie: {result['strategy']}")
            
            eval_data["question"].append(result["question"])
            eval_data["contexts"].append(result["contexts"])
            eval_data["answer"].append(result["answer"])
            eval_data["ground_truth"].append(result["ground_truth"])
            eval_data["id"].append(test_q.get("id"))
            eval_data["language"].append(test_q.get("language", "unknown"))
            metadata["categories"].append(test_q.get("category", "unknown"))
            eval_data["duration"].append(q_duration)
            eval_data["strategy"].append(result["strategy"])  # ✅ Tracking stratégie

        eval_end_time = time.time()
        total_eval_duration = eval_end_time - eval_start_time  # <-- temps total
        dataset = Dataset.from_dict(eval_data)
        
        metrics = [
            faithfulness,
            context_precision,
            context_recall,
            answer_correctness
        ]
        
        config_mistral_safe = RunConfig(
            timeout=600,
            max_retries=20,
            max_wait=120,
            max_workers=1,
            log_tenacity=True
            )



        logging.info("Calcul des métriques RAGAS...")
        
        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                run_config=config_mistral_safe
            )
            
            results_df = results.to_pandas()
            results_df["category"] = metadata["categories"]
            results_df["duration"] = eval_data["duration"]
            results_df["strategy"] = eval_data["strategy"]  # ✅ Ajout colonne stratégie
            
            category_stats = results_df.groupby("category").agg({
                "faithfulness": ["mean"],
                "context_precision": ["mean"],
                "context_recall": ["mean"],
                "answer_correctness": ["mean"]
            }).round(3)

            #  Statistiques par stratégie de routing
            strategy_stats = results_df.groupby("strategy").agg({
                "faithfulness": ["mean", "count"],
                "context_precision": ["mean"],
                "context_recall": ["mean"],
                "answer_correctness": ["mean"],
                "duration": ["mean"]
            }).round(3)
            
            self._save_results(results_df, category_stats, strategy_stats, total_eval_duration)
            
            logging.info("Évaluation terminée avec succès!")
            
            return {
                "results_df": results_df,
                "category_stats": category_stats,
                "strategy_stats": strategy_stats,
                "overall_scores": results_df[
                    ["faithfulness", "context_precision", 
                     "context_recall", "answer_correctness"]
                ].mean().to_dict(),
                "total_eval_duration_sec": total_eval_duration
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de l'évaluation RAGAS: {e}")
            raise
    
    def _save_results(
        self,
        results_df: pd.DataFrame,
        category_stats: pd.DataFrame,
        strategy_stats: pd.DataFrame,
        total_eval_duration: float
    ):
        """Sauvegarde les résultats de l'évaluation dans Ragas_eval/evaluation_results."""

        logging.info("=" * 50)
        logging.info(f"⏱️ Temps total d'exécution de l'évaluation : {total_eval_duration:.2f} sec")
        logging.info("=" * 50)

      
        # Construire le chemin vers Ragas_eval/evaluation_results

        output_path = Root_path / "Ragas_eval" / "evaluation_results"
        output_path.mkdir(parents=True, exist_ok=True)  # crée le dossier si inexistant
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV des résultats détaillés
        results_file = output_path / f"hybrid_evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        logging.info(f" Résultats détaillés sauvegardés: {results_file}")

        # CSV des statistiques par catégorie
        stats_file = output_path / f"hybrid_category_stats_{timestamp}.csv"
        category_stats.to_csv(stats_file)
        logging.info(f" Stats par catégorie: {stats_file}")

        # CSV des statistiques par stratégie
        strategy_file = output_path / f"hybrid_strategy_stats_{timestamp}.csv"
        strategy_stats.to_csv(strategy_file)
        logging.info(f"Stats par stratégie: {strategy_file}")

        # Résumé JSON
        
        summary = {
            "timestamp": timestamp,
            "model": MODEL_NAME,
            "evaluation_type": "HYBRID (RAG + SQL + Router)",
            "num_questions": len(results_df),
            "overall_scores": results_df[
                ["faithfulness", "context_precision", 
                 "context_recall", "answer_correctness"]
            ].mean().to_dict(),
            "category_breakdown": {str(k): v for k, v in category_stats.to_dict().items()},
            "strategy_breakdown": {str(k): v for k, v in strategy_stats.to_dict().items()},
             "avg_duration_sec": results_df["duration"].mean(),
             "total_eval_duration_sec": total_eval_duration,  # <-- temps total ajouté
             "strategy_distribution": results_df["strategy"].value_counts().to_dict()
    }

        summary_file = output_path / f"hybrid_evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logging.info(f" Résumé sauvegardé : {summary_file}")
        
        logging.info("\n" + "="*50)
        logging.info("RÉSUMÉ DES RÉSULTATS HYBRIDES")
        logging.info("="*50)
        logging.info("\n SCORES GLOBAUX:")
        for metric, score in summary["overall_scores"].items():
            logging.info(f"  {metric:25s}: {score:.3f}")
        logging.info("\n DISTRIBUTION DES STRATÉGIES:")
        for strategy, count in summary["strategy_distribution"].items():
            logging.info(f"  {strategy:15s}: {count} questions")
        
        logging.info("="*60)
        


def main():
    """Fonction principale pour exécuter l'évaluation hybride."""
    try:
        logging.info(" Initialisation de l'évaluateur HYBRIDE...")
        evaluator = Evaluator()
        logging.info(" Lancement de l'évaluation...")
        results = evaluator.run_evaluation()
        
        print("\n" + "="*60)
        print("SCORES GLOBAUX (SYSTÈME HYBRIDE)")
        print("="*60)
        for metric, score in results["overall_scores"].items():
            print(f"{metric:25s}: {score:.3f}")
        print("="*60)
        print(" STATISTIQUES PAR STRATÉGIE")
        print("="*60)
        print(results["strategy_stats"])
        print("\n" + "="*60)
        print(f"⏱️ Temps total: {results['total_eval_duration_sec']:.2f} sec")
        print("📁 Résultats sauvegardés dans 'Ragas_eval/evaluation_results/'")
        print("="*60)
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution de l'évaluation: {e}")
        raise


if __name__ == "__main__":
    main()