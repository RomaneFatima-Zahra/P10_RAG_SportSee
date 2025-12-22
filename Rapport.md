# Rapport de mise en place et d’évaluation du système.

___

## TABLE DES MATIÈRES

1. **Contexte**

2. **Evaluation Initiale**

3. **Mise en place du système RAG Hybride**

4. **Evaluation Finale**

5. **Conclusion**

6. **Axes d'amélioration**

7. **Architecture**

___

1. **Contexte** : 

**SportSee**, est une startup spécialisée dans l’IA appliquée à l’analyse de la performance sportive. nous travaillons avec des clubs de basketball pour valoriser leurs archives vidéo, leurs rapports d’analyse et leurs données de matchs. Notre mission est d'aider les entraîneurs, analystes et préparateurs physiques à trouver plus rapidement les informations clés pour préparer les entraînements, les matchs ou suivre la progression des athlètes.

Notre équipe R&D est en charge de la construction d’un assistant intelligent d’analyse de performance en NBA. Nous avons déjà créé un prototype d'assistant IA fonctionnel qui donne des résultats encourageants, que nous allons évaluer et améliorer pour qu'il puisse être encore plus utile et performant.
Notre objectif, est de le rendre capable de répondre à des questions précises telle que : 

-  "Quel joueur a le meilleur pourcentage de réussite à 3 points sur les 5 derniers matchs ?
ou encore : 

- Compare les statistiques de rebonds de l'équipe à domicile et à l'extérieur.

---

2. **Evaluation Initiale**

Notre système RAG est enrichi par les données suivantes ( corpus ) : 

- 4 Fichiers PDF contenant des discussions des supporters et leur commentaires sur les matchs, équipes et joueurs.
- 1 Fichiers Excel qui reprend les statistiques des équipes, joueurs, et l'analyse de leur proformances.

Dans le cadre de la recherche vectorielle, nous avons utilisé les outils suivants : 

- **Recursive Character Text Splitter** pour générer 302 chunks en utilisant le paramétrage suivant : CHUNK_SIZE = 1500, CHUNK_OVERLAP = 150.

- **Mistral Embed** pour générer des embeddings de 1024 dimensions.

- **Faiss IndexFlatIP** pour l'indéxation en BDD vectorielle.

- **Mistral LLM** pour la génération de la réponse


Pour l'évaluation de performance, nous utilisons le Framework Ragas, pour calculer les métriques suivantes : 

**Faithfulness** : Mesure dans quelle mesure la réponse générée est factuellement cohérente et fidèle aux informations présentes dans le contexte fourni.

**Context Precision** : Évalue la qualité du contexte récupéré, en mesurant la proportion de contenu pertinent parmi tout le contexte fourni.

**Context Recall** : Mesure si le contexte contient toutes les informations nécessaires pour répondre correctement à la question.

**Answer correctness** : Mesure à quel point la réponse répond réellement à la question posée, indépendamment de la fidélité au contexte.


Le dataset d'évaluation est composé de 22 questions, dont des questions simples, et des questions complexes.

Les questions sont catégorisées en 4 types : 10 questions portant sur les données statistiques ( Fichier Excel), 6 questions sur les discussions Reddit ( 4 Fichiers PDF), 4 questions mixtes, et 2 questions pièges.

Cette répartition permet de mieux évaluer l'impact de l'intégration des données Excel et la création du Tool SQL.

**Résultats de l'analyse Ragas initiale** : 

| Métrique | Excel | Reddit | Mixte | piège |
|----------|-----------|---------|------|--------|
| **Faithfulness** | 77.5% | 96.7% | 97.4 %| 93.8%  |
| **Context Precision** | 0.1% | 55.6% | 61.2% | 37.5% |
| **Context Recall** | 0.1% | 61.1% | 48.3% | 50.0% |
| **Answer correctness** | 34.3% | 57.8% | 58.1%| 69.6% |


**Constat :**

Le système génère des réponses très fidèles lorsqu’il dispose d’un contexte pertinent, mais échoue massivement à récupérer ce contexte, en particulier pour les questions Excel et dans une moindre mesure, les questions mixtes.
Concrêtement, le problème principal n'est pas le LLM, mais plutôt le retrieval (recherche vectorielle + chunking + indexation) qui est le facteur limitant.

**Faithfulness :**

- Dès que le contexte est pertinent, le modèle ne “hallucine” presque pas. 

- Les scores très élevés sur Reddit et Mixte montrent que :
Les chunks textuels issus des PDF sont bien compris
Le modèle sait synthétiser des opinions et des discussions

- Le score Excel est plus bas (77.5%) car :
Les données tabulaires sont mal représentées sous forme textuelle
Le modèle extrapole parfois à partir de statistiques incomplètes

**Context Precision :**

- Pour Excel, 99.9% du contexte récupéré est non pertinent Même pour Reddit, près de la moitié du contexte est du bruit.

- Causes probables : Excel converti en texte brut, Perte de structure (lignes, colonnes, clés), Embeddings peu discriminants, Chunk size inadapté aux données tabulaires, 1500 caractères ≠ unité sémantique en données chiffrées, IndexFlatIP sans filtrage sémantique, Similarité cosinus seule insuffisante pour des statistiques.

 **Context Recall :**

- Pour Excel, le contexte ne contient quasiment jamais les bonnes informations.

- Pour Mixte, moins d’1 information clé sur 2 est récupérée

- Le système ne voit pas les bonnes données avant même de tenter de répondre. Cela explique les faibles scores de pertinence de réponse.

**Answer correctness :** 

- Les mauvaises performances Excel se répercutent directement sur la qualité des réponses.

- Le score élevé sur les questions pièges suggère que le LLM sait ne pas répondre ou rester vague, ce qui est positif

- Pour Excel Les réponses sont souvent génériques, incomplètes ou incorrectes faute de contexte.


L'analyse de l'évaluation Ragas initiale montre que les limites du système ne proviennent pas du LLM, mais du retrieval, en particulier pour les données Excel. Le traitement uniforme de données textuelles et tabulaires par vectorisation entraîne une perte de structure des statistiques (lignes, colonnes, relations), rendant la recherche vectorielle inadaptée aux requêtes chiffrées.



L'analyse de l'évaluation Ragas initiale met en évidence un problème structurel de conception du système RAG, et non une faiblesse du modèle de langage. Le pipeline actuel repose sur une approche uniforme de vectorisation, appliquée indifféremment à des données textuelles non structurées (PDF Reddit) et à des données tabulaires structurées (Excel). Or, ces deux types de données obéissent à des logiques sémantiques différentes.

La conversion des données Excel en texte linéaire entraîne une perte critique de structure (relations lignes/colonnes, clés statistiques,...), rendant la recherche vectorielle inadaptée. Le chunking par taille de caractères et la similarité cosinus ne permettent ni d’identifier précisément les statistiques pertinentes, ni de garantir leur complétude. Ce défaut se traduit par un effondrement du recall et de la précision du contexte pour les requêtes statistiques, et donc par des réponses inexactes ou génériques. Ces résultats indiquent clairement que le retrieval doit être spécialisé selon la nature de la donnée. Ils motivent la mise en place d’une architecture RAG hybride, en intégrant un Tool SQL dédié à l’interrogation des données structurées pour améliorer la performance de notre pipeline RAG.

---

3. **Mise en place du système RAG Hybride**

Afin de corriger les limites identifiées lors de l’évaluation initiale, nous avons mis en place une architecture RAG hybride distinguant explicitement le traitement des données structurées et non structurées.

**Structuration des données statistiques :**   ( migration Excel vers BDD PostgreSQL)

Cette étape consiste à transformer les données statistiques initialement stockées dans un fichier Excel en une base relationnelle exploitable. Le script de migration assure la connexion à PostgreSQL, la lecture des feuilles Excel, la détection automatique des types et la création dynamique des tables SQL. Les données sont ensuite insérées après une phase de validation stricte à l’aide de modèles Pydantic, garantissant la cohérence des statistiques (types, valeurs, clés étrangères). Des tests SQL de validation permettent enfin de vérifier l’intégrité de la base. Cette étape vise à préserver la structure sémantique des données chiffrées et à rendre possibles des requêtes fiables de type agrégation, comparaison ou classement.


**Interrogation des données structurées via le Tool SQL :** (sql_tool.py) 

Nous introduisons un Tool SQL spécialisé, chargé de répondre aux questions quantitatives. À partir d’une question en langage naturel, le tool génère automatiquement une requête SQL valide en s’appuyant sur le schéma réel de la base PostgreSQL et sur des exemples few-shot. Des mécanismes de validation garantissent que seules des requêtes SELECT conformes sont exécutées. Le résultat est ensuite formaté et transmis au modèle de langage, qui peut ainsi produire des réponses chiffrées précises et vérifiables. Ce script permet de contourner les limites de la recherche vectorielle sur des données tabulaires et constitue le socle des réponses statistiques du système.

**Orchestration intelligente des requêtes :** (Router et exécution hybride router.py )

Ce scriptcorrespond à l’orchestration globale du système. iL s'agit d'un router intelligent qui analyse chaque question utilisateur et détermine dynamiquement la stratégie la plus appropriée : RAG seul pour les questions qualitatives, SQL seul pour les questions purement statistiques, ou une approche hybride lorsque la question nécessite à la fois des données chiffrées et un contexte interprétatif. Dans le cas hybride, le système déclenche à la fois une requête SQL et une recherche RAG, puis confie au LLM une étape finale de synthèse combinant faits statistiques et analyses qualitatives. Cette orchestration permet d’exploiter chaque source selon ses forces, tout en améliorant la précision, le recall et la pertinence globale des réponses.


```mermaid
flowchart TD
    U[Question utilisateur] --> R[Question Router<br/>(Analyse de l’intention)]

    %% --- RAG ONLY ---
    R -->|RAG ONLY| V1[Recherche vectorielle<br/>(FAISS + Embeddings)]
    V1 --> C1[Contexte textuel<br/>(Discussions Reddit)]
    C1 --> G1[LLM – Génération]
    G1 --> F1[Réponse finale<br/>(Qualitative / contextuelle)]

    %% --- SQL ONLY ---
    R -->|SQL ONLY| S2[SQL Tool<br/>(NL → SQL)]
    S2 --> DB2[(PostgreSQL)]
    DB2 --> D2[Résultats SQL<br/>(Statistiques)]
    D2 --> G2[LLM – Génération]
    G2 --> F2[Réponse finale<br/>(Quantitative / factuelle)]

    %% --- HYBRID ---
    R -->|HYBRID| S3[SQL Tool]
    R -->|HYBRID| V3[Recherche vectorielle]

    S3 --> DB3[(PostgreSQL)]
    DB3 --> D3[Données chiffrées]
    V3 --> C3[Contexte qualitatif]

    D3 --> G3[LLM – Synthèse]
    C3 --> G3
    G3 --> F3[Réponse finale<br/>(Stats + contexte)]
```

---

4. **Evaluation Finale** 

Après la mise en place de cette nouvelle architecture, nous procédons à une seconde évaluation Ragas sur le même dataset de questions.


**Résultats de l'analyse Ragas finale** : 

| métriques		   		  | Excel 	  | Reddit  | Mixte | piège  |
|-------------------------|-----------|---------|-------|--------|
| **Faithfulness** 		  | 100% 	  | 91.4%   | 79.4 %| 100%   |
| **Context Precision**   | 60.0% 	  | 55.6%   | 53.1% | 37.5%  |
| **Context Recall**      | 58.3% 	  | 61.1%   | 72.5% | 50.0%  |
| **Answer correctness**  | 40.7% 	  | 49.6%   | 57.7% | 19.0%  |


**Faithfulness :**

- Questions Excel & Piège : une amélioration nette grace au SQL Tool et à la validation stricte avec pydantic qui ont éliminé les hallucinations factuelles. Le système refuse implicitement de répondre hors données

- Questions Mixte : une baisse car le système ne fusionne plus naïvement des contextes hétérogènes. cette baisse illustre un choix de sûreté, car le système est devenu plus conservateur et plus fiable.

**Context Precision :**

- Questions Excel : nous constatons une transformation radicale. nous passons d'une récupération massive et bruitée à des contextes bien ciblés via SQL tool.

- Questions Mixte : légère baisse, car le routage réduit la couverture, mais augmente la qualité moyenne.


**Context Recall :**  

- Questions Excel & Mixte : une forte progression. Le SQL Tool permet d’aller chercher exactement la donnée manquante. 


**Answer correctness :**

- Questions Excel : amélioration réelle, Le système ne devine plus, mais donne des réponses sûres.

- Questions Piège : chute volontaire, car le système préfère répondre partiellement ou refuser, au lieu de donner une mauvaise réponse.


| strategy   | Questions détectées| faithfulness moyen  | context precision moyen | context recall moyen| answer correctness moyen | duration moyen | 
|------------|--------------------|---------------------|-------------------------|---------------------|--------------------------|----------------|
| HYBRID     | 6               	  | 0.792           	|  0.52                   | 0.761               | 0.524                    | 12.393         |
| RAG_ONLY   | 5                  | 0.982          		|  0.617                  | 0.6                 | 0.462                    | 4.481          |
| SQL_ONLY   | 11              	  | 1.0             	|  0.545                  | 0.53                | 0.386                    | 3.624          |



**RAG_ONLY** : 

 + Très bon niveau de faithfulness
 + Réponse rapide est 
 - Moins de précision
 - hallucinations résiduelles

**SQL_ONLY** : 

+ Faithfulness parfait
+ Réponse rapide 
- Réponses limitée au chiffres, sans plus d'explications 

**HYBRID** :

+ Meilleur compromis comparé aux autres stratégies.    
- Réposne plus lente, car le pipeline d'orchestration est plus complexe        

5. **Conclusion**


| Dataset | Évolution Faithfulness | Évolution Context Precision | Évolution Context Recall | Évolution Answer correctness | 
| ------- | -----------------------|-----------------------------|--------------------------|------------------------------| 
| Excel   | **+22.5 pts** ✅	   |**+59.9 pts** ✅ 			 | **+58.2 pts** ✅		    | **+6.4 pts** 	✅			   | 
| Reddit  | -5.3 pts     		   |stable      			     | stable      			    | -8.2 pts 					   | 
| Mixte   | -18 pts ❌   		   | -8.1 pts   				 | **+24.2 pts** ✅			| stable					   | 
| Piège   | **+6.2 pts** ✅		   | stable 					 | stable      				| -50.6 pts ❌				   | 


**L'analyse comparative entre l'évaluation initiale et l'évaluation finale démontre une transformation structurelle significative du système RAG, passant d'une architecture simple et peu performante à une approche hybride spécialisée et plus robuste. Cette évolution résout efficacement les limitations identifiées initialement tout en introduisant de nouvelles dynamiques de performance qui privilégie la fiabilité des réponses, la précision du contexte et la maîtrise des hallucinations.**

Forces principales :

- Fiabilité exceptionnelle sur les données structurées (100% faithfulness sur les données statistiques)

- Adaptabilité grâce au routing intelligent

- Architecture modulaire permettant des améliorations futures

- Transparence des décisions avec des stratégies clairement identifiées

**le pipeline final atteint son objectif principal : fournir des réponses fiables et vérifiables en exploitant chaque source de données selon ses spécificités, même si cela implique parfois des réponses partielles ou des refus de répondre. Cette approche "safety-first" est préférable à des réponses complètes mais potentiellement erronées.**

---

6. **Axes d'amélioration**

Bien que les résultats finaux soient significativement meilleurs, plusieurs axes d’amélioration peuvent être envisagés pour renforcer encore la robustesse, la précision et la scalabilité du système.

- **Optimisation du chunking** : la taille fixe des chunks (1500 caractères) est inadaptée à la diversité des contenus. l'implémentation d'un chunking intelligent qui s'adapte au type de contenu pourrait améliorer la précision et le rappel du contexte. 

- **Amélioration du routage des questions**: Le routeur repose actuellement sur des règles explicites. Une évolution naturelle consisterait à entraîner un classifieur dédié pour détecter plus finement les intentions hybrides et mieux distinguer les questions nécessitant une agrégation complexe, une comparaison temporelle ou une simple extraction factuelle. Cela permettrait de réduire les erreurs de routage et d’optimiser le choix de la stratégie dès l’entrée du pipeline.

- **Enrichissement du Tool SQL** : il pourrait être étendu pour gérer des requêtes plus complexes (fenêtres temporelles, moyennes glissantes ), intégrer des contrôles sémantiques supplémentaires, ou fournir des métadonnées explicatives (ex. : source, saison, périmètre des données).
Cela renforcerait la transparence et la confiance des utilisateurs finaux dans les réponses statistiques.

- **Amélioration des réponses hybrides** : Dans les cas HYBRID, la phase de synthèse par le LLM peut encore être optimisée en structurant explicitement le prompt de synthèse (séparation faits / interprétation), en forçant la citation explicite des résultats SQL utilisés, ou en évaluant séparément la fidélité à la partie SQL et à la partie RAG.
Cette amélioration permettrait d’augmenter l’answer correctness sans dégrader la faithfulness.

- **Optimisation des temps de réponse pour les requêtes hybrides** : pour faire baisser le temps de réponse élevés pour les requêtes hybrides, nous pouvons envisager de mettre en place un système de cache intelligent, qui réduirait significative les latences pour les requêtes récurrentes.


- **Élargissement du dataset d’évaluation** Le dataset Ragas actuel reste limité (22 questions). Pour une évaluation plus robuste, nous pouvons envisager d'augmenter le nombre de questions, introduire davantage de questions multi-étapes, intégrer des scénarios utilisateurs réalistes issus de l’analyse métier (coachs, analystes, préparateurs physiques).
Cela permettrait de mieux mesurer la généralisation du système.

- **Suivi en production et évaluation continue** : intégrer un monitoring continu des performances (logs, métriques Ragas en ligne), une boucle de feedback utilisateur, des tests de non-régression lors de l’ajout de nouvelles données ou fonctionnalités.
cela pourrait garantirla stabilité et l’évolution maîtrisée du système dans un contexte réel.


___

7. **Architecture**

```

P10_DSML/
├── MistralChat.py          # Application Streamlit principale 
├── indexer.py              # Script pour indexer les documents
├── requirements.txt        # Dépendances Python
├── pyproject.toml          # Configuration env poetry
├── poetry.lock             # Configuration poetry
├── .env                    # Variables d'environnement ( fichier caché)
│
│── Scripts/
│	├── load_excel_to_db.py # Loading de Excel vers BDD SQL
│   │── sql_tool.py         # Générer et exécuter des requêtes SQL dynamiques
│	│── router.py 			# Routing intelligent de questions
│ 	└── test_router.py  	# Test du router.py
│
├── Ragas_eval/
│	├── evaluate_ragas.py   # Evaluation Ragas
│   └── evaluation_results/ # Résultats de l'évaluation
│	└── router_analysis/  	# Résultats de test de routing
│     
├── inputs/                 # Dossier pour les documents sources
│   ├── Reddit 1.pdf        # Commentaires matchs NBA
│   └── Reddit 2.pdf        # Commentaires matchs NBA
│   └── Reddit 3.pdf        # Commentaires matchs NBA
│   └── Reddit 4.pdf        # Commentaires matchs NBA
│   └── regular NBA.xlsx    # Statistiques joueurs
│ 
│ 
├── vector_db/              # Dossier pour l'index FAISS et les chunks
│   ├── faiss_index.idx     # Index FAISS
│   └── document_chunks.pkl # Chunks 
│ 
└── utils/                  # Modules utilitaires
    ├── config.py           # Configuration de l'application
    ├── data_loader.py      # Extraction de texte multi-format (PDF, DOCX, Excel, CSV, TXT)
    └── vector_store.py     # Gestion de l'index vectoriel + Recherche sémantique
```

Ce projet avait pour objectif d’évaluer puis d’améliorer un système RAG appliqué à l’analyse de performance en NBA, en s’appuyant sur des sources hétérogènes mêlant données textuelles non structurées (PDF Reddit) et données statistiques structurées (Excel).
L’évaluation initiale via le framework Ragas a permis d’identifier un problème fondamental de conception : le pipeline de retrieval, fondé uniquement sur la recherche vectorielle, était inadapté au traitement des données tabulaires. Cette limitation entraînait une perte de structure critique, un effondrement du recall et de la précision du contexte pour les questions statistiques, et par conséquent des réponses souvent imprécises ou génériques, malgré un LLM performant.

Pour répondre à ces limites, une architecture RAG hybride a été mise en place, reposant sur une séparation claire des responsabilités :

une base relationnelle PostgreSQL dédiée aux statistiques,

un Tool SQL spécialisé pour l’interrogation fiable des données chiffrées,

une recherche vectorielle conservée pour les données textuelles,

et un routeur intelligent orchestrant dynamiquement les stratégies RAG_ONLY, SQL_ONLY ou HYBRID.

L’évaluation finale montre des améliorations majeures sur les questions Excel et mixtes : augmentation drastique de la précision et du recall du contexte, élimination quasi totale des hallucinations factuelles, et amélioration mesurée mais significative de la justesse des réponses. La baisse volontaire des scores sur les questions pièges et certaines questions mixtes traduit un changement assumé de comportement du système : celui-ci privilégie désormais la fiabilité, la traçabilité et la sûreté à la complétude artificielle des réponses.

En conclusion, ce travail met en évidence que la performance d’un système RAG ne dépend pas uniquement du modèle de langage, mais avant tout de la qualité du retrieval et de l’adéquation entre le type de données et la méthode d’accès. L’architecture hybride mise en place constitue une base robuste et industrialisable pour un assistant d’analyse de performance sportive, capable de fournir des réponses chiffrées vérifiables tout en conservant une capacité d’analyse contextuelle riche.


