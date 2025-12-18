"""
Script de Migration de Excel vers PostgreSQL
1 : Configuration et connexion à la base de données
2 : Lecture Excel et détection types
3 : CRÉATION DES TABLES SQL
4 : INSERTION DES DONNÉES + validation pydantic
5 : TESTS DE VALIDATION SQL
"""

import sys
from pathlib import Path
import psycopg2
from psycopg2 import sql
import pandas as pd
import numpy as np
from pydantic import BaseModel, field_validator, ConfigDict
from typing import Optional


# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Import de la configuration depuis utils/config.py
from utils.config import DB_CONFIG

# =============================================================================
#  CONFIGURATION Pydantic
# =============================================================================

class PlayerStatModel(BaseModel):
    player: str
    team: str
    pts: Optional[float]
    gp: Optional[int]

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True
    )

    @field_validator("team")
    @classmethod
    def validate_team(cls, v):
        if v is None or len(v) != 3:
            raise ValueError("Code équipe invalide")
        return v.upper()

    @field_validator("pts")
    @classmethod
    def validate_pts(cls, v):
        if v is not None and v < 0:
            raise ValueError("PTS négatif")
        return v


# =============================================================================
# BLOC 1 : CONFIGURATION ET CONNEXION POSTGRESQL
# =============================================================================

def test_connection():
    """
    Teste la connexion à PostgreSQL et affiche les informations de la base
    """
    print("=" * 80)
    print("BLOC 1 : TEST DE CONNEXION POSTGRESQL")
    print("=" * 80)
    
    # Vérifier que le mot de passe est rempli
    if not DB_CONFIG['password']:
        print("❌ ERREUR : Le mot de passe n'est pas rempli dans DB_CONFIG")
        print("   → Ouvre le fichier et remplis DB_CONFIG['password']")
        sys.exit(1)
    
    try:
        # Connexion à PostgreSQL
        print(f"\n1. Tentative de connexion à {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
        conn = psycopg2.connect(**DB_CONFIG)
        print("   ✅ Connexion réussie !")
        
        # Créer un curseur
        cursor = conn.cursor()
        
        # Vérifier la version PostgreSQL
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"\n2. Version PostgreSQL :")
        print(f"   {version}")
        
        # Vérifier la base de données actuelle
        cursor.execute("SELECT current_database();")
        current_db = cursor.fetchone()[0]
        print(f"\n3. Base de données connectée : {current_db}")
        
        # Lister les tables existantes (devrait être vide pour une nouvelle base)
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        tables = cursor.fetchall()
        
        if tables:
            print(f"\n4. Tables existantes ({len(tables)}) :")
            for table in tables:
                print(f"   - {table[0]}")
        else:
            print("\n4. Tables existantes : Aucune (base vide) ✅")
        
        # Fermer la connexion
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 80)
        print("✅ BLOC 1 VALIDÉ : Connexion PostgreSQL opérationnelle")
        print("=" * 80)
        return True
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ ERREUR DE CONNEXION :")
        print(f"   {e}")
        print("\n   Vérifications à faire :")
        print("   1. PostgreSQL est-il démarré ?")
        print("   2. Le mot de passe est-il correct ?")
        print("   3. La base 'nba_stats' existe-t-elle dans pgAdmin ?")
        return False
        
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE : {e}")
        return False


# =============================================================================
# EXÉCUTION DU BLOC 1
# =============================================================================

if __name__ == "__main__":
    test_connection()


# =============================================================================
# BLOC 2 : LECTURE EXCEL ET DÉTECTION TYPES
# =============================================================================

def load_excel_data():
    """
    Charge les données Excel et détecte les types automatiquement
    """
    print("\n" + "=" * 80)
    print("BLOC 2 : LECTURE EXCEL ET DÉTECTION TYPES")
    print("=" * 80)
    
    # Chemins des fichiers
    excel_file = root_dir / "inputs" / "regular_NBA.xlsx"
    dict_file = root_dir / "inputs" / "dictionnaire_enrichi.csv"
    
    # Vérifier que les fichiers existent
    if not excel_file.exists():
        print(f"❌ ERREUR : Fichier Excel introuvable : {excel_file}")
        return None, None, None
    
    print(f"\n1. Chargement du fichier Excel...")
    print(f"   Chemin : {excel_file}")
    
    try:
        # Charger les 3 feuilles nécessaires
        df_nba = pd.read_excel(excel_file, sheet_name='Données NBA', header=1)
        df_equipe = pd.read_excel(excel_file, sheet_name='Equipe')
        
        # Nettoyer les colonnes vides (Unnamed)
        df_nba = df_nba.loc[:, ~df_nba.columns.astype(str).str.startswith('Unnamed')]
        
        print(f"   ✅ Feuille 'Données NBA' : {df_nba.shape[0]} joueurs × {df_nba.shape[1]} colonnes")
        print(f"   ✅ Feuille 'Equipe' : {df_equipe.shape[0]} équipes")
        
        # Charger le dictionnaire enrichi
        if dict_file.exists():
            df_dict = pd.read_csv(dict_file)
            print(f"   ✅ Dictionnaire enrichi : {len(df_dict)} mappings")
        else:
            print(f"   ⚠️  Dictionnaire enrichi introuvable : {dict_file}")
            df_dict = None
        
        # Analyser les types détectés par pandas
        print(f"\n2. Analyse des types détectés par pandas...")
        print(f"\n   Types de colonnes (échantillon) :")
        
        type_counts = df_nba.dtypes.value_counts()
        for dtype, count in type_counts.items():
            print(f"   - {dtype} : {count} colonnes")
        
        # Afficher quelques exemples
        print(f"\n   Exemples de colonnes par type :")
        for dtype in type_counts.index[:3]:
            cols = df_nba.select_dtypes(include=[dtype]).columns[:3].tolist()
            print(f"   - {dtype} : {', '.join(cols)}")
        
        print("\n" + "=" * 80)
        print("✅ BLOC 2 VALIDÉ : Données Excel chargées et types détectés")
        print("=" * 80)
        
        return df_nba, df_equipe, df_dict
        
    except Exception as e:
        print(f"\n❌ ERREUR lors du chargement Excel : {e}")
        return None, None, None
    

# =============================================================================
# BLOC 3 : CRÉATION DES TABLES SQL
# =============================================================================

def create_tables(df_nba, df_equipe, df_dict):
    """
    Crée les tables player_stats et team_codes dans PostgreSQL
    """
    print("\n" + "=" * 80)
    print("BLOC 3 : CRÉATION DES TABLES SQL")
    print("=" * 80)
    
    try:
        # Connexion à PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 1. CRÉER LA TABLE team_codes (référence pour foreign key)
        print("\n1. Création de la table 'team_codes'...")
        
        cursor.execute("""
            DROP TABLE IF EXISTS team_codes CASCADE;
            CREATE TABLE team_codes (
                code VARCHAR(3) PRIMARY KEY,
                nom_complet VARCHAR(100) NOT NULL
            );
        """)
        print("   ✅ Table 'team_codes' créée")
        
        # 2. CRÉER LA TABLE player_stats avec mapping des types
        print("\n2. Création de la table 'player_stats'...")
        
        # Mapping pandas → PostgreSQL
        type_mapping = {
            'int64': 'INTEGER',
            'float64': 'FLOAT',
            'object': 'VARCHAR(255)'
        }
        
        # Construire le mapping des noms de colonnes
        name_mapping = {}
        
        # Si dictionnaire disponible, utiliser les noms normalisés
        if df_dict is not None:
            # Créer un mapping depuis le dictionnaire
            for excel_name, sql_name in zip(df_dict['nom_excel'], df_dict['nom_colonne_sql']):
                name_mapping[str(excel_name)] = sql_name
        
        # Ajouter/normaliser les colonnes du dataframe
        for col in df_nba.columns:
            col_str = str(col)
            
            # Si pas dans le dictionnaire, normaliser automatiquement
            if col_str not in name_mapping:
                normalized = col_str.lower().replace('%', '_pct').replace('/', '_').replace(' ', '_').replace('+', 'plus').replace('-', '_minus').replace(':', '_')
                
                # Si le nom commence par un chiffre, ajouter préfixe "col_"
                if normalized[0].isdigit():
                    normalized = 'col_' + normalized
                
                name_mapping[col_str] = normalized
        
        # Construire la liste des colonnes SQL avec leurs types
        columns_sql = []
        for col in df_nba.columns:
            col_str = str(col)
            pandas_type = str(df_nba[col].dtype)
            sql_type = type_mapping.get(pandas_type, 'VARCHAR(255)')
            
            # Utiliser le nom normalisé
            sql_col_name = name_mapping[col_str]
            
            columns_sql.append(f"{sql_col_name} {sql_type}")
        
        # Ajouter la contrainte de clé étrangère pour team
        columns_definition = ",\n            ".join(columns_sql)
        
        create_table_sql = f"""
            DROP TABLE IF EXISTS player_stats CASCADE;
            CREATE TABLE player_stats (
                {columns_definition},
                FOREIGN KEY (team) REFERENCES team_codes(code)
            );
        """
        
        cursor.execute(create_table_sql)
        print(f"   ✅ Table 'player_stats' créée avec {len(columns_sql)} colonnes")
        
        # Afficher quelques exemples de colonnes créées
        print("\n3. Exemples de colonnes créées :")
        for i, (excel_col, sql_col) in enumerate(list(name_mapping.items())[:5], 1):
            excel_col_str = str(excel_col)
            # Accéder à la colonne avec la clé originale
            if excel_col in df_nba.columns:
                pandas_type = str(df_nba[excel_col].dtype)
            else:
                pandas_type = 'unknown'
            sql_type = type_mapping.get(pandas_type, 'VARCHAR(255)')
            print(f"   {i}. {excel_col_str:15s} → {sql_col:20s} ({sql_type})")
        print(f"   ... et {len(name_mapping) - 5} autres colonnes")
        
        # Commit et fermeture
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 80)
        print("✅ BLOC 3 VALIDÉ : Tables SQL créées avec succès")
        print("=" * 80)
        
        return name_mapping
        
    except Exception as e:
        print(f"\n❌ ERREUR lors de la création des tables : {e}")
        if conn:
            conn.rollback()
        return None

# =============================================================================
# BLOC 4 : INSERTION DES DONNÉES
# =============================================================================

def insert_data(df_nba, df_equipe, name_mapping):
    """
    Insère les données dans les tables PostgreSQL
    """
    print("\n" + "=" * 80)
    print("BLOC 4 : INSERTION DES DONNÉES")
    print("=" * 80)
    
    try:
        # Connexion à PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # 1. INSÉRER LES DONNÉES DE team_codes
        print("\n1. Insertion des données dans 'team_codes'...")
        
        for _, row in df_equipe.iterrows():
            cursor.execute("""
                INSERT INTO team_codes (code, nom_complet)
                VALUES (%s, %s)
            """, (row['Code'], row["Nom complet de l'équipe"]))
        
        print(f"   ✅ {len(df_equipe)} équipes insérées")
        conn.commit()  # ← AJOUTE CETTE LIGNE
        print("   ✅ Commit effectué pour team_codes")
        
        # 2. INSÉRER LES DONNÉES DE player_stats
        print("\n2. Insertion des données dans 'player_stats'...")
        
        # Renommer les colonnes du dataframe selon le mapping
        df_nba_renamed = df_nba.copy()
        df_nba_renamed.columns = [name_mapping.get(str(col), str(col).lower()) for col in df_nba.columns]
        
        # Remplacer les NaN par None (NULL en SQL)
        df_nba_renamed = df_nba_renamed.where(pd.notna(df_nba_renamed), None)
        
        # Remplacer les NaN par None (NULL en SQL)
        df_nba_renamed = df_nba_renamed.where(pd.notna(df_nba_renamed), None)

        # =============================
        # VALIDATION PYDANTIC (AJOUT)
        # =============================
        validated_rows = []
        rejected = 0

        for i, row in df_nba_renamed.iterrows():
            row_dict = row.to_dict()
            try:
                PlayerStatModel(
                    player=row_dict.get("player"),
                    team=row_dict.get("team"),
                    pts=row_dict.get("pts"),
                    gp=row_dict.get("gp"),
                    )
        # Si OK → on garde TOUTE la ligne
                validated_rows.append(row_dict)
            except Exception as e:
                rejected += 1
                print(f"⚠️ Ligne {i} rejetée : {e}")

        df_nba_renamed = pd.DataFrame(validated_rows)
        print(f"   ➜ {len(df_nba_renamed)} lignes validées")
        print(f"   ➜ {rejected} lignes rejetées")

        # =============================
        # FIN AJOUT PYDANTIC
        # =============================

        # Utiliser pandas.to_sql pour insertion rapide
        from sqlalchemy import create_engine
        
        # Créer l'engine SQLAlchemy
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        
        # Insérer les données (append pour ajouter, pas replace)
        df_nba_renamed.to_sql(
            'player_stats',
            engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=100)
        
        print(f"   ✅ {len(df_nba_renamed)} joueurs insérés")
        
        # Commit et fermeture
        conn.commit()
        print("   ✅ Commit effectué pour player_stats")
        cursor.close()
        conn.close()
        engine.dispose()
        
        print("\n" + "=" * 80)
        print("✅ BLOC 4 VALIDÉ : Données insérées avec succès")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'insertion des données : {e}")
        if conn:
            conn.rollback()
        return False
    
# =============================================================================
# BLOC 5 : TESTS DE VALIDATION SQL
# =============================================================================

def validate_data():
    """
    Valide que les données ont été correctement insérées avec des requêtes SQL
    """
    print("\n" + "=" * 80)
    print("BLOC 5 : TESTS DE VALIDATION SQL")
    print("=" * 80)
    
    try:
        # Connexion à PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # TEST 1 : Compter les équipes
        print("\n1. Test COUNT sur team_codes...")
        cursor.execute("SELECT COUNT(*) FROM team_codes;")
        count_teams = cursor.fetchone()[0]
        print(f"   ✅ {count_teams} équipes dans la base")
        assert count_teams == 30, f"Erreur : attendu 30 équipes, trouvé {count_teams}"
        
        # TEST 2 : Compter les joueurs
        print("\n2. Test COUNT sur player_stats...")
        cursor.execute("SELECT COUNT(*) FROM player_stats;")
        count_players = cursor.fetchone()[0]
        print(f"   ✅ {count_players} joueurs dans la base")
        assert count_players == 569, f"Erreur : attendu 569 joueurs, trouvé {count_players}"
        
        # TEST 3 : Vérifier un joueur spécifique
        print("\n3. Test SELECT avec filtre...")
        cursor.execute("""
            SELECT player, team, pts 
            FROM player_stats 
            WHERE player = 'Shai Gilgeous-Alexander';
        """)
        result = cursor.fetchone()
        if result:
            print(f"   ✅ Joueur trouvé : {result[0]} ({result[1]}) - {result[2]} PTS")
        else:
            print("   ⚠️  Joueur 'Shai Gilgeous-Alexander' non trouvé")
        
        # TEST 4 : Top 5 players
        print("\n4. Test ORDER BY - Top 5 players...")
        cursor.execute("""
            SELECT player, team, pts 
            FROM player_stats 
            ORDER BY pts DESC 
            LIMIT 5;
        """)
        top_scorers = cursor.fetchall()
        print("   Top 5 players :")
        for i, (player, team, pts) in enumerate(top_scorers, 1):
            print(f"   {i}. {player:30s} ({team}) - {pts} PTS")
        
        # TEST 5 : Jointure team_codes + player_stats
        print("\n5. Test JOIN - Joueurs avec nom complet équipe...")
        cursor.execute("""
            SELECT p.player, t.nom_complet, p.pts 
            FROM player_stats p
            JOIN team_codes t ON p.team = t.code
            ORDER BY p.pts DESC
            LIMIT 3;
        """)
        join_results = cursor.fetchall()
        print("   Top 3 avec nom complet équipe :")
        for player, team_name, pts in join_results:
            print(f"   - {player:30s} ({team_name:30s}) - {pts} PTS")
        
        # Fermeture
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 80)
        print("✅ BLOC 5 VALIDÉ : Toutes les validations SQL sont ")
        print("=" * 80)
        print("\n🎉 LOADING EXCEL VERS SQL TERMINÉ AVEC SUCCÈS !")
        print("   Base de données PostgreSQL 'nba_stats' créée et enrichie.")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR lors de la validation : {e}")
        return False
    
# =============================================================================
# EXÉCUTION DES BLOCS
# =============================================================================

if __name__ == "__main__":
    # BLOC 1
    if not test_connection():
        sys.exit(1)
    
    # BLOC 2
    df_nba, df_equipe, df_dict = load_excel_data()
    if df_nba is None:
        sys.exit(1)
    
    # BLOC 3 
    name_mapping = create_tables(df_nba, df_equipe, df_dict)
    if name_mapping is None:
        sys.exit(1)

    # BLOC 4 
    if not insert_data(df_nba, df_equipe, name_mapping):
        sys.exit(1)

    # BLOC 5 
    if not validate_data():
        sys.exit(1)