"""
Script de test du router sur le dataset RAGAS.
Analyse comment le router classe chaque question (SQL, RAG, Hybrid)
et génère des statistiques détaillées.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
from collections import Counter
import pandas as pd
from dotenv import load_dotenv

# Setup paths
Root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(Root_path))

from utils.config import MISTRAL_API_KEY
from scripts.router import QuestionRouter, RoutingDecision
from mistralai.client import MistralClient

# --------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------

load_dotenv()
Dataset_path = Root_path / "Ragas_eval" / "rag_dataset.json"
Output_path = Root_path / "Ragas_eval" / "router_analysis"
Output_path.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# ANALYSIS CLASS
# --------------------------------------------------------------------

class RouterAnalyzer:
    """Analyse le comportement du router sur le dataset RAGAS"""
    
    def __init__(self):
        """Initialise l'analyseur avec le router"""
        self.mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
        self.router = QuestionRouter(self.mistral_client)
        logger.info(" RouterAnalyzer initialisé")
    
    def load_dataset(self, json_path: Path = Dataset_path) -> List[Dict]:
        """Charge le dataset RAGAS"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("Le fichier JSON doit contenir une liste")
            
            logger.info(f"📚 {len(data)} questions chargées depuis {json_path}")
            return data
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            raise
    
    def analyze_routing(self, dataset: List[Dict]) -> pd.DataFrame:
        """
        Analyse le routing de toutes les questions du dataset.
        
        Returns:
            DataFrame avec les résultats d'analyse
        """
        logger.info("🔍 Début de l'analyse du routing...")
        
        results = []
        
        for i, item in enumerate(dataset, 1):
            question = item["question"]
            category = item.get("category", "unknown")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{len(dataset)}] Question: {question[:80]}...")
            
            try:
                # Router la question
                decision = self.router.route_question(question)
                
                # Construire le résultat
                result = {
                    "id": item.get("id"),
                    "question": question,
                    "category": category,
                    "language": item.get("language", "unknown"),
                    "strategy": decision.strategy,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning,
                    "sql_subquestion": decision.sql_subquestion,
                    "rag_subquestion": decision.rag_subquestion
                }
                
                results.append(result)
                
                # Log du résultat
                logger.info(f" Stratégie: {decision.strategy.upper()}")
                logger.info(f"   Confiance: {decision.confidence:.2%}")
                logger.info(f"   Raisonnement: {decision.reasoning}")
                
                if decision.strategy == "hybrid":
                    logger.info(f"   SQL Sub-Q: {decision.sql_subquestion}")
                    logger.info(f"   RAG Sub-Q: {decision.rag_subquestion}")
                
            except Exception as e:
                logger.error(f"❌ Erreur pour la question {i}: {e}")
                results.append({
                    "id": item.get("id"),
                    "question": question,
                    "category": category,
                    "language": item.get("language", "unknown"),
                    "strategy": "ERROR",
                    "confidence": 0.0,
                    "reasoning": f"Erreur: {str(e)}",
                    "sql_subquestion": None,
                    "rag_subquestion": None
                })
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ Analyse terminée!")
        
        return pd.DataFrame(results)
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """Génère des statistiques sur les résultats de routing"""
        
        category_dist = (
            df.groupby("category")["strategy"]
            .value_counts()
            .unstack(fill_value=0)
            .to_dict(orient="index")
            )

        stats = {
            "total_questions": len(df),
            "strategy_distribution": df["strategy"].value_counts().to_dict(),
            "strategy_percentages": (df["strategy"].value_counts(normalize=True) * 100)
                                     .round(2).to_dict(),
            "avg_confidence_by_strategy": (df.groupby("strategy")["confidence"]
                                           .mean().round(3).to_dict()),
            "category_distribution": category_dist,  # ✅ JSON-safe
            "low_confidence_questions": int((df["confidence"] < 0.6).sum()),
            "high_confidence_questions": int((df["confidence"] >= 0.8).sum())
            }

        
        return stats
    
    def save_results(self, df: pd.DataFrame, stats: Dict):
        """Sauvegarde les résultats de l'analyse"""
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. CSV détaillé
        csv_file = Output_path / f"router_analysis_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f" CSV sauvegardé: {csv_file}")
        
        # 2. JSON des statistiques
        json_file = Output_path / f"router_stats_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"📊 Stats JSON sauvegardées: {json_file}")
        
        
def main():
    """Fonction principale"""
    
    print("\n🚀 ANALYSE DU ROUTER SUR LE DATASET RAGAS")
    print("="*80)
    
    try:
        # 1. Initialiser l'analyseur
        analyzer = RouterAnalyzer()
        
        # 2. Charger le dataset
        dataset = analyzer.load_dataset()
        
        # 3. Analyser le routing
        results_df = analyzer.analyze_routing(dataset)
        
        # 4. Générer les statistiques
        stats = analyzer.generate_statistics(results_df)
        
        # 5. Sauvegarder les résultats
        analyzer.save_results(results_df, stats)
        
        print("\n✅ Analyse terminée avec succès!")
        print(f" Résultats disponibles dans: {Output_path}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution: {e}")
        raise


if __name__ == "__main__":
    main()