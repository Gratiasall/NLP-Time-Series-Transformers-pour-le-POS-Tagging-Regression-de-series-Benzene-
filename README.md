🎯 ***Objectif du repo***

Ce dépôt rassemble deux Travaux Pratiques (TP) complets — chacun avec énoncé et solution — couvrant deux domaines clés du machine learning :

1. NLP :

Désambiguïsation nom vs verbe (jeu de données noun-verb de Google Research)

Part-of-Speech Tagging (POS) sur données CoNLL (via pyconll)

Modèles LSTM/GRU et Transformers, métriques de séquence, préparation DataLoader, entraînement et analyse d’erreurs.

2. Séries temporelles (Time Series) :

Régression de la concentration de benzène (jeu Benzene Concentration – Zenodo)

Chargement/structuration avec aeon, features temporelles, normalisation, séparation temporelle stricte, entraînement PyTorch Lightning et MSE/RMSE.

📂 ***Contenu***

**TP_3_1_énoncé.ipynb** — Prédiction de la concentration en benzène (énoncé)
Formulation du problème, périmètre des features, consignes d’EDA, préparation des jeux temporels (train/val/test), objectifs de métriques (MSE/RMSE).

**TP_3_1_ALLAKONON_VODOUMBO_solution.ipynb** — Prédiction de la concentration en benzène (solution)
Pipeline complet : ingestion avec aeon, features et normalisation, DataLoader, modèle PyTorch (régression), entraînement Lightning, suivi des pertes, visualisations et interprétation des résultats.

**TP_3_2_énoncé.ipynb** — NLP : désambiguïsation nom/verbe & POS tagging (énoncé)
Tâches séquentielles, préparation des données CoNLL (pyconll), définition des architectures (RNN/GRU/LSTM/Transformer), métriques séquence (accuracy/f1/recall selon la partie), et attentes d’analyse d’erreurs.

**TP_3_2_solution.ipynb** — NLP : désambiguïsation nom/verbe & POS tagging (solution)
Implémentations LSTM/GRU et Transformer : tokenisation, embeddings, masquage, DataLoader séquentiel, boucle d’entraînement Lightning, torchmetrics pour le suivi, matrices de confusion/exemples mal classés, commentaires sur les compromis biais/variance.

🧰 ***Stack & bibliothèques***

- Python 3 (Jupyter Notebooks)

- PyTorch & PyTorch Lightning (boucles d’entraînement propres et reproductibles)

- torchmetrics, torchinfo (métriques et résumé de modèle)

- aeon (séries temporelles : chargement/transformations)

- pyconll (parsing CoNLL pour POS)

- pandas, numpy, matplotlib.

📊 ***Résultats & insights***

- Benzène : les variables temporelles et la structure d’agrégation améliorent significativement la RMSE vs baseline naïve; les diagnostics montrent l’importance de la saisonnalité et de la variabilité intra-jour/semaine.

- NLP : le Transformer capture mieux les dépendances longues que les RNN classiques sur le POS tagging; l’analyse d’erreurs met en évidence les ambiguïtés lexicales récurrentes (homographes nom/verbe).