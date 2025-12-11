# evaluate_ragas.py
"""
Script d'évaluation du système RAG basé sur le framework RAGAS.
Mesure la pertinence, la fidélité et la cohérence des réponses.
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

# ✅ LangChain Mistral (UNIQUE STACK)
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.messages import HumanMessage


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

class RAGEvaluator:
    """Classe pour évaluer le système RAG avec RAGAS."""
    
    def __init__(self):
        """Initialise l'évaluateur avec le Vector Store et le client Mistral."""
        self.vector_store = VectorStoreManager()

        # ✅ LLM LangChain
        self.llm = ChatMistralAI(
            api_key=MISTRAL_API_KEY,
            model=MODEL_NAME,
            temperature=0.1,
            max_retries=3,
            timeout=30

        )


        # ✅ Embeddings LangChain
        self.embeddings = MistralAIEmbeddings(
            api_key=MISTRAL_API_KEY,
            model=EMBEDDING_MODEL,
            max_retries=3,
            timeout=30

        )

        # Vérifier que l'index est chargé
        if self.vector_store.index is None or not self.vector_store.document_chunks:
            raise ValueError("Le Vector Store n'est pas initialisé. Exécutez 'python indexer.py' d'abord.")
        
        logging.info(f"RAGEvaluator initialisé avec {self.vector_store.index.ntotal} vecteurs.")
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Génère une réponse à partir de la question et du contexte.
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
        """Récupère le contexte pertinent pour une question."""
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

        contexts, sources = self.retrieve_context(question)
        context_str = "\n\n".join(contexts)
        answer = self.generate_answer(question, context_str)
        
        return {
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth if ground_truth else "Non fourni"
        }

    # --------------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------------
    def run_evaluation(
        self,
        test_questions: Optional[List[Dict]] = None
    ) -> Dict:

        if test_questions is None:
            test_questions = self.load_test_questions_from_json(Dataset_path)

        logging.info(f"Début de l'évaluation avec {len(test_questions)} questions...")
        
        eval_data = {
            "question": [],
            "contexts": [],
            "answer": [],
            "ground_truth": [],
            "id": [],
            "language": []
        }
        
        metadata = {
            "categories": [],
            "sources": []
        }
        
        for i, test_q in enumerate(test_questions, 1):
            logging.info(f"Évaluation question {i}/{len(test_questions)}: {test_q['question'][:60]}...")
             
            # Ajouter un délai pour éviter le rate limiting
            if i > 1:
                time.sleep(2)

            result = self.evaluate_single_question(
                question=test_q["question"],
                ground_truth=test_q.get("ground_truth")
            )
            
            eval_data["question"].append(result["question"])
            eval_data["contexts"].append(result["contexts"])
            eval_data["answer"].append(result["answer"])
            eval_data["ground_truth"].append(result["ground_truth"])
            eval_data["id"].append(test_q.get("id"))
            eval_data["language"].append(test_q.get("language", "unknown"))
            metadata["categories"].append(test_q.get("category", "unknown"))
        
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
            
            category_stats = results_df.groupby("category").agg({
                "faithfulness": ["mean"],
                "context_precision": ["mean"],
                "context_recall": ["mean"],
                "answer_correctness": ["mean"]
            }).round(3)
            
            self._save_results(results_df, category_stats)
            
            logging.info("Évaluation terminée avec succès!")
            
            return {
                "results_df": results_df,
                "category_stats": category_stats,
                "overall_scores": results_df[
                    ["faithfulness", "context_precision", 
                     "context_recall", "answer_correctness"]
                ].mean().to_dict()
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de l'évaluation RAGAS: {e}")
            raise
    
    def _save_results(
        self,
        results_df: pd.DataFrame,
        category_stats: pd.DataFrame,
    ):
        """Sauvegarde les résultats de l'évaluation dans Ragas_eval/evaluation_results."""
        # Construire le chemin vers Ragas_eval/evaluation_results

        output_path = Root_path / "Ragas_eval" / "1st_evaluation_results"
        output_path.mkdir(parents=True, exist_ok=True)  # crée le dossier si inexistant
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV des résultats détaillés
        results_file = output_path / f"evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        logging.info(f"Résultats détaillés sauvegardés: {results_file}")

        # CSV des statistiques par catégorie
        
        stats_file = output_path / f"category_stats_{timestamp}.csv"
        category_stats.to_csv(stats_file)
        logging.info(f"Statistiques par catégorie sauvegardées: {stats_file}")

        # Résumé JSON
        
        summary = {
            "timestamp": timestamp,
            "model": MODEL_NAME,
            "num_questions": len(results_df),
            "overall_scores": results_df[
                ["faithfulness", "context_precision", 
                 "context_recall", "answer_correctness"]
            ].mean().to_dict(),
            "category_breakdown": {str(k): v for k, v in category_stats.to_dict().items()},
    }
        
        summary_file = output_path / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logging.info(f"Résumé sauvegardé: {summary_file}")
        
        logging.info("\n" + "="*50)
        logging.info("RÉSUMÉ DES RÉSULTATS")
        logging.info("="*50)
        for metric, score in summary["overall_scores"].items():
            logging.info(f"{metric}: {score:.3f}")
        logging.info("="*50)


def main():
    """Fonction principale pour exécuter l'évaluation."""
    try:
        evaluator = RAGEvaluator()
        results = evaluator.run_evaluation()
        
        print("\n" + "="*60)
        print("SCORES GLOBAUX")
        print("="*60)
        for metric, score in results["overall_scores"].items():
            print(f"{metric:25s}: {score:.3f}")
        print("="*60)
        
        print("\nLes résultats détaillés ont été sauvegardés dans '1st_evaluation_results/'")
        
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution de l'évaluation: {e}")
        raise


if __name__ == "__main__":
    main()