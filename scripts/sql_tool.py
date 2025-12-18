# sql_tool.py
"""
Tool SQL LangChain pour générer et exécuter des requêtes SQL dynamiques.
Permet au LLM de répondre à des questions quantitatives sur les données NBA.
"""

import logging
from typing import List, Dict, Any
import psycopg2
import pandas as pd
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

from pathlib import Path
import sys

from sqlalchemy import create_engine

# =======================
# PYDANTIC
# =======================

from pydantic import BaseModel, Field, field_validator


class QuestionModel(BaseModel):
    question: str = Field(..., min_length=5)

    @field_validator("question")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("La question ne peut pas être vide")
        return v


class SQLQueryModel(BaseModel):
    query: str = Field(..., min_length=10)

    @field_validator("query")
    @classmethod
    def must_be_select(cls, v: str) -> str:
        if not v.strip().lower().startswith("select"):
            raise ValueError("Seules les requêtes SELECT sont autorisées")
        return v


class SQLResultModel(BaseModel):
    success: bool
    row_count: int
    columns: List[str]
    data: List[Dict[str, Any]]
    formatted_data: str
    query: str


# =======================
# PATH & CONFIG
# =======================

root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from utils.config import DB_CONFIG


# =======================
# LOGGING
# =======================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =======================
# DATABASE TOOL
# =======================

class NBADatabaseTool:
    """Tool pour interagir avec la base de données NBA via SQL"""

    def __init__(self):
        self.db_config = DB_CONFIG
        self._validate_connection()
        self.schema_info = self._get_schema_info()

        # SQLAlchemy engine (corrige le warning pandas)
        self.engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )

    def _validate_connection(self):
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            logging.info("✅ Connexion à la base de données NBA validée")
        except Exception as e:
            logging.error(f"❌ Erreur de connexion à la base: {e}")
            raise

    def _get_schema_info(self) -> str:
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
            """)

            schema = cursor.fetchall()
            cursor.close()
            conn.close()

            schema_dict = {}
            for table, column, dtype in schema:
                schema_dict.setdefault(table, []).append(f"{column} ({dtype})")

            schema_text = "SCHÉMA DE LA BASE DE DONNÉES NBA:\n\n"
            for table, columns in schema_dict.items():
                schema_text += f"Table: {table}\n"
                schema_text += "Colonnes: " + ", ".join(columns[:10])
                if len(columns) > 10:
                    schema_text += f", ... (+{len(columns)-10} autres)"
                schema_text += "\n\n"

            return schema_text

        except Exception as e:
            logging.error(f"Erreur lors de la récupération du schéma: {e}")
            return "Schéma non disponible"

    def execute_sql(self, query: str) -> Dict[str, Any]:
        try:
            # Validation Pydantic de la requête SQL
            validated_query = SQLQueryModel(query=query).query

            logging.info(f"Exécution de la requête SQL: {validated_query[:100]}...")

            from sqlalchemy import text
            with self.engine.connect() as conn:
                df = pd.read_sql(text(validated_query), conn)

            logging.info(f"✅ Requête exécutée: {len(df)} lignes retournées")

            results = SQLResultModel(
                success=True,
                row_count=len(df),
                columns=df.columns.tolist(),
                data=df.to_dict("records"),
                formatted_data=self._format_results(df),
                query=validated_query
            ).model_dump()

            return results

        except Exception as e:
            logging.error(f"❌ Erreur SQL: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    def _format_results(self, df: pd.DataFrame, max_rows: int = 10) -> str:
        if df.empty:
            return "Aucun résultat trouvé."

        df_display = df.head(max_rows)
        formatted = df_display.to_string(index=False)

        if len(df) > max_rows:
            formatted += f"\n... et {len(df) - max_rows} autres lignes"

        return formatted

# =======================
# SQL GENERATOR
# =======================   


class SQLQueryGenerator:
    """Génère des requêtes SQL à partir de questions en langage naturel"""
    
    # Exemples few-shot pour améliorer la génération
    FEW_SHOT_EXAMPLES = """
EXEMPLES DE QUESTIONS ET REQUÊTES SQL:

Question: "Quel joueur a le meilleur pourcentage de tirs à 3 points cette saison ?"
SQL: SELECT player, team, three_p_pct FROM player_stats WHERE col_3pa > 50 ORDER BY three_p_pct DESC LIMIT 1;

Question: "Combien de points LeBron James a-t-il marqué cette saison ?"
SQL: SELECT player, pts FROM player_stats WHERE player LIKE '%LeBron James%';

Question: "Quels sont les 5 meilleurs scoreurs ?"
SQL: SELECT player, team, pts FROM player_stats ORDER BY pts DESC LIMIT 5;

Question: "Quelle est la moyenne de points des Lakers ?"
SQL: SELECT AVG(pts) as moyenne_points FROM player_stats WHERE team = 'LAL';

Question: "Compare les rebonds moyens des équipes Lakers et Celtics"
SQL: SELECT team, AVG(trb) as avg_rebounds FROM player_stats WHERE team IN ('LAL', 'BOS') GROUP BY team;

Question: "Quels joueurs ont plus de 25 points par match ?"
SQL: SELECT player, team, pts FROM player_stats WHERE pts > 25 ORDER BY pts DESC;

Question: "Quel est le pourcentage de tirs moyen de Stephen Curry ?"
SQL: SELECT player, fg_pct, three_p_pct FROM player_stats WHERE player LIKE '%Stephen Curry%';

Question: "Combien de joueurs ont marqué plus de 20 points ?"
SQL: SELECT COUNT(*) as nombre_joueurs FROM player_stats WHERE pts > 20;

Question: "Liste des meneurs de jeu avec plus de 8 passes décisives"
SQL: SELECT player, team, ast FROM player_stats WHERE pos LIKE '%PG%' AND ast > 8 ORDER BY ast DESC;

Question: "Quelle équipe a le plus de joueurs qui marquent plus de 15 points ?"
SQL: SELECT team, COUNT(*) as nb_scoreurs FROM player_stats WHERE pts > 15 GROUP BY team ORDER BY nb_scoreurs DESC LIMIT 1;
"""
    
    PROMPT_TEMPLATE = """Tu es un expert SQL spécialisé dans les bases de données statistiques NBA. 
    Ta tâche est de convertir une question en langage naturel en une requête SQL valide.

#### Colonnes principales de `player_stats`:
- player : nom du joueur (VARCHAR)
- team : code de l'équipe (VARCHAR, clé étrangère vers team_codes.code)
- pts : points par match (FLOAT)
- gp : matchs joués (INTEGER)
autres colonnes : 
wins, losses, minutes,
pts, fgm, fga, fg_pct,
three_pa, three_p_pct,
ftm, fta, ft_pct,
oreb, dreb, reb,
ast, tov, stl, blk, pf,
fp, dd2, td3, plus__minus,
off_rtg, def_rtg, net_rtg,
ast_pct, ast_to_ratio, ast_ratio,
oreb_pct, dreb_pct, reb_pct,
to_ratio, efg_pct, ts_pct, usg_pct,
pace, pie, poss

N’utilise AUCUNE autre colonne.

#### Colonnes de `team_codes`:
- code : code équipe (VARCHAR, PRIMARY KEY)
- nom_complet : nom complet de l'équipe (VARCHAR)

Tu DOIS utiliser uniquement ces colonnes SQL exactes :


{schema_info}
{few_shot_examples}

INSTRUCTIONS IMPORTANTES:
1. Génère UNIQUEMENT la requête SQL, sans explications
2. Utilise les noms de colonnes exacts du schéma (ex: 'pts' pas 'points', 'three_p_pct' pas '3pt%')
3. Pour les codes d'équipes, utilise les codes à 3 lettres (LAL, BOS, etc.)
4. Pour les recherches de noms, utilise LIKE avec % (ex: WHERE player LIKE '%LeBron%')
5. Limite les résultats avec LIMIT si nécessaire
6. Pour lesagrégations, utilise AVG(), SUM(), COUNT(), etc.
   Pour les moyennes AVEC PostgreSQL, tu dois TOUJOURS écrire : ROUND(AVG(colonne)::numeric, 2) et pas ROUND(AVG(colonne), 2)
7. La table principale est 'player_stats'
8. Colonnes importantes: player, team, pts, ast, trb, fg_pct, three_p_pct, pos, gp
9. Utilise TOUJOURS la jointure correcte entre `player_stats.team` et `team_codes.code`
5. Trie les résultats par ordre décroissant pour les "top", "meilleur", "plus"
6. Pour les questions sur les équipes, utilise la jointure avec team_codes

### Exemples de questions et requêtes SQL correspondantes:

Question: "Quels sont les 5 meilleurs scoreurs de la saison ?"
Requête SQL:
```sql
SELECT player, team, pts 
FROM player_stats 
ORDER BY pts DESC 
LIMIT 5;

Question: "Quelle est la moyenne de points par équipe ?"
Requête SQL:
SELECT t.nom_complet, ROUND(AVG(p.pts), 2) as moyenne_points
FROM player_stats p
JOIN team_codes t ON p.team = t.code
GROUP BY t.nom_complet
ORDER BY moyenne_points DESC;

Question: "Quels joueurs des Lakers ont joué plus de 60 matchs ?"
Requête SQL:
SELECT p.player, p.gp, p.pts
FROM player_stats p
JOIN team_codes t ON p.team = t.code
WHERE t.nom_complet LIKE '%Lakers%' AND p.gp > 60
ORDER BY p.pts DESC;

QUESTION: {question}

REQUÊTE SQL:"""
    
    def __init__(self, schema_info: str):
        """
        Initialise le générateur de requêtes.
        
        Args:
            schema_info: Information sur le schéma de la base
        """
        self.schema_info = schema_info
        self.prompt = PromptTemplate(
            input_variables=["schema_info", "few_shot_examples", "question"],
            template=self.PROMPT_TEMPLATE
        )
    
    def generate_sql(self, question: str, llm_client) -> str:
        """
        Génère une requête SQL à partir d'une question.
        
        Args:
            question: Question en langage naturel
            llm_client: Client LLM (Mistral) pour la génération
            
        Returns:
            Requête SQL générée
        """
        # Construire le prompt avec les exemples few-shot
        prompt_text = self.prompt.format(
            schema_info=self.schema_info,
            few_shot_examples=self.FEW_SHOT_EXAMPLES,
            question=question
        )
        
        logging.info(f"Génération SQL pour: {question}")
        
        try:
            # Appeler le LLM pour générer la requête
            from mistralai.models.chat_completion import ChatMessage
            
            messages = [ChatMessage(role="user", content=prompt_text)]
            
            response = llm_client.chat(
                model="mistral-large-latest",
                messages=messages,
                temperature=0.1,  # Température basse pour plus de précision
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Nettoyer la requête (enlever les markdown, commentaires, etc.)
            sql_query = self._clean_sql(sql_query)
            
            logging.info(f"SQL généré: {sql_query}")
            
            return SQLQueryModel(query=sql_query).query
            
        except Exception as e:
            logging.error(f"Erreur lors de la génération SQL: {e}")
            raise
    
    def _clean_sql(self, sql: str) -> str:
        """Nettoie la requête SQL générée"""
        # Enlever les markdown code blocks
        sql = sql.replace("```sql", "").replace("```", "")
        
        # Enlever les commentaires
        lines = sql.split('\n')
        sql = ' '.join(line for line in lines if not line.strip().startswith('--'))
        
        # Enlever les espaces multiples
        sql = ' '.join(sql.split())
        
        # Ajouter le point-virgule si manquant
        if not sql.endswith(';'):
            sql += ';'
        
        return sql

# =======================
# MAIN TOOL
# =======================

class NBADataTool:
    """
    Tool principal qui combine la génération et l'exécution de requêtes SQL.
    Compatible avec LangChain.
    """
    
    def __init__(self, llm_client):
        """
        Initialise le tool.
        
        Args:
            llm_client: Client Mistral pour la génération SQL
        """
        self.db_tool = NBADatabaseTool()
        self.sql_generator = SQLQueryGenerator(self.db_tool.schema_info)
        self.llm_client = llm_client
    
    def run(self, question: str) -> str:
        """
        Exécute le tool: génère SQL puis exécute la requête.
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            Résultats formatés en texte
        """
        try:
            # 1. Générer la requête SQL
            question = QuestionModel(question=question).question
            sql_query = self.sql_generator.generate_sql(question, self.llm_client)
            
            # 2. Exécuter la requête
            results = self.db_tool.execute_sql(sql_query)
            
            # 3. Formater la réponse
            if results["success"]:
                response = f"📊 RÉSULTATS DE LA BASE DE DONNÉES:\n\n"
                response += f"Requête SQL exécutée:\n{results['query']}\n\n"
                response += f"Nombre de résultats: {results['row_count']}\n\n"
                response += "Données:\n"
                response += results['formatted_data']
                
                return response
            else:
                return f"❌ Erreur lors de l'exécution SQL: {results['error']}"
                
        except Exception as e:
            logging.error(f"Erreur dans le tool NBA: {e}")
            return f"❌ Erreur: {str(e)}"
    
    def is_quantitative_question(self, question: str) -> bool:
        """
        Détecte si une question nécessite des données quantitatives.
        
        Args:
            question: Question de l'utilisateur
            
        Returns:
            True si la question est quantitative
        """
        # Mots-clés indiquant une question quantitative
        quantitative_keywords = [ # Mots-clés statistiques : 
            'combien', 'nombre', 'moyenne', 'total', 'meilleur', 'pire',
            'top', 'classement', 'statistiques', 'stats', 'pourcentage',
            'points', 'rebonds', 'passes', 'comparaison', 'compare',
            'plus que', 'moins que', 'supérieur', 'inférieur',
            'combien de', 'quel joueur', 'quelle équipe', 'qui a',
            'liste', 'tous les', 'quels sont','count', 'somme', 'sum',
            'moyenne', 'moyen', 'average', 'avg', 'mean',
            'maximum', 'max', 'minimum', 'min', 'ranking', 'pire',
            'plus de', 'moins de', 'données', 'chiffres', 'quel est', 'quelle est',
            'quelles sont','qui a', 'qui sont','par équipe', 'par joueur', 'par match',
            'tous les joueurs', 'toutes les équipes','comparer', 'différence', 'écart',
            'contres', 'interceptions','%', 'efficacité', 'rating'
            ]
        
        
        question_lower = question.lower()
        
        # Vérifier la présence de mots-clés
        return any(keyword in question_lower for keyword in quantitative_keywords)

# =======================
# LANGCHAIN TOOL
# =======================

def create_langchain_tool(llm_client) -> Tool:
    """
    Crée un Tool LangChain à partir du NBADataTool.
    
    Args:
        llm_client: Client Mistral
        
    Returns:
        Tool LangChain prêt à être utilisé
    """
    nba_tool = NBADataTool(llm_client)
    
    return Tool(
        name="NBA_Database_Query",
        func=nba_tool.run,
        description="""Utile pour répondre à des questions quantitatives sur les statistiques NBA.
        Utilise ce tool quand l'utilisateur demande:
        - Des statistiques de joueurs (points, rebonds, passes, etc.)
        - Des comparaisons entre joueurs ou équipes
        - Des classements ou top N joueurs
        - Des moyennes, totaux ou pourcentages
        - Des informations chiffrées sur les performances
        
        Exemples: "Combien de points LeBron a marqué?", "Top 5 scoreurs", 
        "Compare les stats de Lakers vs Celtics", "Moyenne de rebounds des pivots"
        """
    )


# =============================================================================
# FONCTION UTILITAIRE POUR STREAMLIT
# =============================================================================

def should_use_sql_tool(question: str) -> bool:
    """
    Détermine si une question nécessite le SQL tool.
    
    Args:
        question: Question de l'utilisateur
        
    Returns:
        True si le SQL tool doit être utilisé
    """
    tool = NBADataTool(None)  # Pas besoin du LLM juste pour détecter
    return tool.is_quantitative_question(question)


def execute_sql_query_standalone(question: str, llm_client) -> str:
    """
    Exécute une requête SQL de manière autonome (sans LangChain).
    Utile pour intégration directe dans Streamlit.
    
    Args:
        question: Question de l'utilisateur
        llm_client: Client Mistral
        
    Returns:
        Résultats formatés
    """
    tool = NBADataTool(llm_client)
    return tool.run(question)


# =============================================================================
# TESTS
# =============================================================================

if __name__ == "__main__":
    print("🧪 Tests du SQL Tool NBA")
    print("=" * 80)
    
    # Test 1: Connexion
    print("\n1. Test de connexion...")
    try:
        db_tool = NBADatabaseTool()
        print("✅ Connexion OK")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        exit(1)
    
    # Test 2: Exécution SQL directe
    print("\n2. Test d'exécution SQL directe...")
    results = db_tool.execute_sql("SELECT player, team, pts FROM player_stats ORDER BY pts DESC LIMIT 5;")
    
    if results["success"]:
        print("✅ Requête exécutée")
        print(f"   Résultats: {results['row_count']} lignes")
        print(f"\n{results['formatted_data']}")
    else:
        print(f"❌ Erreur: {results['error']}")
    
    # Test 3: Détection de questions quantitatives
    print("\n3. Test de détection de questions...")
    
    test_questions = [
        ("Combien de points LeBron a marqué ?", True),
        ("Qui est le meilleur joueur de basket ?", False),
        ("Top 5 des scoreurs", True),
        ("Raconte-moi l'histoire des Lakers", False),
        ("Compare les stats de Lakers et Celtics", True),
    ]
    
    tool = NBADataTool(None)
    for question, expected in test_questions:
        result = tool.is_quantitative_question(question)
        status = "✅" if result == expected else "❌"
        print(f"   {status} '{question}' → {result}")
    
    print("\n" + "=" * 80)
    print("✅ Tests terminés")