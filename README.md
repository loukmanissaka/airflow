 Projet Airflow : Prévision Météo et Sélection du Meilleur Modèle

Ce projet utilise Apache Airflow pour orchestrer un pipeline de traitement de données météo. Il intègre la collecte des données depuis l’API OpenWeatherMap, le prétraitement, l’entraînement de plusieurs modèles de régression, et la sélection automatique du meilleur modèle basé sur la MAE (Mean Absolute Error).
Objectifs
-Collecter automatiquement les données météorologiques (température, humidité, pression...).
-Nettoyer et transformer les données pour l'entraînement.
-Entraîner plusieurs modèles (Régression Linéaire, Arbre de Décision, Forêt Aléatoire).
-Sélectionner le meilleur modèle selon la performance (MAE la plus proche de zéro).
-Sauvegarder ce modèle pour une utilisation future.

Technologies utilisées
*Python, Apache Airflow, Pandas, Scikit-learn, API OpenWeatherMap, Docker (optionnel), Git/GitHub

Structure du projet

.
├── dags/
│   └── eval_ISSAKA.py          # Le DAG principal
├── raw_files/                  # Données brutes collectées
├── clean_data/                 # Données nettoyées + meilleur modèle

 Fonctionnement du pipeline
Extraction : Récupération des données météo depuis OpenWeatherMap.
Transformation : Nettoyage et formatage des données pour l'apprentissage.
Training : Entraînement de trois modèles différents.
Évaluation : Calcul de la MAE et sélection automatique du meilleur.
Sauvegarde : Export du modèle sélectionné au format .pckl.

Voici un exemple de résultat obtenue

Résumé des scores des modèles :
Modèle               Score (MAE)
------------------------------
Linear Regression    -0.6497
Decision Tree        -85.4918
Random Forest        -38.0510

Meilleur modèle sélectionné : Linear Regression
