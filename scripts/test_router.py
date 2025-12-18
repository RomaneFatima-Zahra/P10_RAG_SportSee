# test_router.py
"""
Tests complets du système de routing et d'exécution hybride.
Permet de valider le fonctionnement des 3 stratégies : RAG, SQL, Hybrid.
"""

import sys
from pathlib import Path

# Ajout du répertoire racine au path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from mistralai.client import MistralClient
from utils.config import MISTRAL_API_KEY
from utils.vector_store import VectorStoreManager
from scripts.sql_tool import NBADataTool
from scripts.router import QuestionRouter, HybridQueryExecutor


def print_separator(char="=", length=80):
    """Affiche un séparateur"""
    print(char * length)


def print_section(title):
    """Affiche un titre de section"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def test_question(executor, router, question):
    """
    Teste une question complète.
    
    Args:
        executor: HybridQueryExecutor
        router: QuestionRouter
        question: Question à tester
    """
    print(f"\n📝 QUESTION: {question}")
    print("-" * 80)
    
    # 1. Routing
    decision = router.route_question(question)
    print(f"✓ Stratégie: {decision.strategy}")
    print(f"✓ Confiance: {decision.confidence:.2f}")
    print(f"✓ Raisonnement: {decision.reasoning}")
    
    if decision.sql_subquestion:
        print(f"✓ SQL sous-question: {decision.sql_subquestion}")
    if decision.rag_subquestion:
        print(f"✓ RAG sous-question: {decision.rag_subquestion}")
    
    # 2. Exécution
    print("\n🔄 Exécution...")
    result = executor.execute(question, decision, search_k=3)
    
    # 3. Affichage du résultat
    print("\n📄 RÉSULTAT:")
    print("-" * 80)
    print(result[:500] + "..." if len(result) > 500 else result)
    print("-" * 80)


def main():
    """Fonction principale de test"""
    print_section("🧪 TESTS DU SYSTÈME DE ROUTING NBA")
    
    # Initialisation
    print("\n 1️⃣ Initialisation des composants...")
    
    try:
        client = MistralClient(api_key=MISTRAL_API_KEY)
        print("✅ Client Mistral OK")
        
        vector_store = VectorStoreManager()
        if vector_store.index is None:
            print("❌ Vector Store non chargé")
            return
        print(f"✅ Vector Store OK ({vector_store.index.ntotal} vecteurs)")
        
        sql_tool = NBADataTool(client)
        print("✅ SQL Tool OK")
        
        router = QuestionRouter(client)
        print("✅ Router OK")
        
        executor = HybridQueryExecutor(client, vector_store, sql_tool)
        print("✅ Hybrid Executor OK")
        
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return
    
    # Questions de test
    test_questions = [
        # RAG Only (qualitatives)
        {
            "category": "RAG Only - Opinion",
            "questions": [
                "Qui est considéré comme le meilleur par les fans ?",
                "Que pensent les fans de la saison actuelle des Miami Heats ?",
                "Y a-t-il des débats sur le meilleur joueur actuel ?"
            ]
        },
        
        # SQL Only (quantitatives)
        {
            "category": "SQL Only - Statistiques",
            "questions": [
                "Combien de points Anthony Edwards a marqué cette saison ?",
                "Quels sont les 5 meilleurs scoreurs de la NBA ?",
                "Quelle est la moyenne de points des Minnesota Timberwolves ?"
            ]
        },
        
        # Hybrid (mixtes)
        {
            "category": "Hybrid - Questions mixtes",
            "questions": [
                "Pourquoi Miami Heats sont meilleurs ? Montre-moi leurs statistiques et les analyses",
                "Compare les performances de Julius Randle et les opinions sur lui",
                "Paolo Banchero est-il toujours au top niveau ? Compare ses stats et ce que disent les fans",
            ]
        }
    ]
    
    # Exécution des tests
    for test_group in test_questions:
        print_section(f"📂 {test_group['category']}")
        
        for question in test_group['questions']:
            try:
                test_question(executor, router, question)
            except Exception as e:
                print(f"❌ Erreur lors du test: {e}")
            
            print("\n" + "="*80 + "\n")
    
    # Statistiques finales
    print_section("📊 RÉSUMÉ DES TESTS")
    print("\n✅ Tests terminés avec succès")
    print(f"✓ {len([q for g in test_questions for q in g['questions']])} questions testées")
    print("\n💡 Vérifiez que:")
    print("  - Les questions qualitatives utilisent RAG")
    print("  - Les questions quantitatives utilisent SQL")
    print("  - Les questions mixtes utilisent HYBRID")
    print("  - Les résultats sont cohérents et pertinents")


if __name__ == "__main__":
    main()