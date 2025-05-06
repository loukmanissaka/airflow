from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime
import os
import json
import requests
from pathlib import Path
from joblib import dump
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

Variable.set("cities", '["paris", "london", "washington"]')

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 1),
    'retries': 1,
}

dag = DAG(
    'open_weather_map_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['weather','datascientest']
)


##Extraction et transformation des données
def get_weather_data():
    """
 Cette fonction recupere les ville de la variable airflow, extrait les données meteo selon les villes et les sauve garde dans un fichier se trouvant à /opt/airflow/raw_files
    """
    API_key = "88ff082e4908e947565e1b0a14f6dcfe" #la clef api
    cities = Variable.get("cities", deserialize_json=True) # recupereation des ville de la variable airflow

    results = {}
    for city in cities:
        response = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_key}"
        )
        results[city] = response.json()
    
    # Définition du répertoire où seront stockés les fichiers bruts
    raw_files_dir = Path("/opt/airflow/raw_files")
    
    raw_files_dir.mkdir(parents=True, exist_ok=True) # Création du dossier s'il n'existe pas (parents=True crée aussi les répertoires parents si nécessaire)
 

    # Sauvegarde des résultats météo dans un fichier JSON
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_path = raw_files_dir / f"{now_str}.json"

    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)

def transform_data_into_csv(n_files=None, filename='data.csv'):
    parent_folder = '/opt/airflow/raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)
    if n_files:
        files = files[:n_files]

    dfs = []

    for f in files:
        filepath = os.path.join(parent_folder, f)
        try:
            with open(filepath, 'r') as file:
                data_temp = json.load(file)
        except json.JSONDecodeError as e:
            print(f"Erreur JSON dans le fichier {f}: {e}")
            continue
        
        if not isinstance(data_temp, dict):
            print(f"Format invalide dans {f} : attendu dict, reçu {type(data_temp).__name__}")
            continue

        for city_name, city_data in data_temp.items():
            try:
                dfs.append({
                    'temperature': city_data['main']['temp'],
                    'city': city_data['name'],
                    'pression': city_data['main']['pressure'],
                    'date': f.split('.')[0]
                })
            except KeyError as e:
                print(f"Clé manquante dans {f} pour {city_name}: {e}")
                continue

    df = pd.DataFrame(dfs)
    print(df.head(10))
    os.makedirs('/opt/airflow/clean_data', exist_ok=True)
    df.to_csv(os.path.join('/opt/airflow/clean_data', filename), index=False)

##Partie machine learning
#calcul du score du modele
def compute_model_score(model, X, y, cv=3):
    """
    Calcule le score moyen d'un modèle via la validation croisée.
    Utilise l'erreur quadratique moyenne négative (RMSE négatif).

    Parameters:
        model: Modèle sklearn à évaluer.
        X: Variables explicatives.
        y: Variable cible.
        cv: Nombre de folds pour la validation croisée.

    Returns:
        float: Score moyen (RMSE négatif).
    """
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring='neg_mean_squared_error'
    )
    mean_score = np.mean(scores)
    return mean_score

#entrainement ddu modele 

def train_and_save_model(model, X, y, path_to_model='/opt/airflow/clean_data/best_model.pckl'):
    """
    Entraîne un modèle sur les données et l'enregistre dans un fichier.

    Parameters:
        model: Modèle sklearn à entraîner.
        X: Features d'entraînement.
        y: Cible d'entraînement.
        path_to_model: Chemin d'enregistrement du modèle (fichier .pckl ou .joblib).

    Returns:
        None
    """
    # Entraînement du modèle
    model.fit(X, y)

    # Création du dossier si nécessaire
    os.makedirs(os.path.dirname(path_to_model), exist_ok=True)

    # Sauvegarde du modèle
    dump(model, path_to_model)
    print(f" Modèle entraîné et sauvegardé à : {path_to_model}")

#

def prepare_data(path_to_data='/opt/airflow/clean_data/fulldata.csv'):
    # Lecture des données
    df = pd.read_csv(path_to_data)

    # Tri par ville et date
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []

    for city in df['city'].unique():
        df_temp = df[df['city'] == city].copy()

        # Création de la cible (target)
        df_temp['target'] = df_temp['temperature'].shift(1)

        # Création des variables explicatives (features)
        for i in range(1, 10):
            df_temp[f'temp_m-{i}'] = df_temp['temperature'].shift(-i)

        # Suppression des lignes avec NaN
        df_temp.dropna(inplace=True)

        dfs.append(df_temp)

    # Fusion des sous-dataframes
    df_final = pd.concat(dfs, axis=0)

    # Suppression de la colonne date
    df_final.drop(columns=['date'], inplace=True)

    # Encodage one-hot de la variable 'city'
    df_final = pd.get_dummies(df_final, columns=['city'])

    # Séparation des features et de la cible
    X = df_final.drop(columns=['target'])
    y = df_final['target']

    return X, y


if __name__ == '__main__':
    X, y = prepare_data('./clean_data/fulldata.csv')

    score_lr = compute_model_score(LinearRegression(), X, y)
    score_dt = compute_model_score(DecisionTreeRegressor(), X, y)

    best_model = LinearRegression() if score_lr < score_dt else DecisionTreeRegressor()

    train_and_save_model(best_model, X, y, '/opt/airflow/clean_data/best_model.pickle')


def train_lr(**context):
    X, y = prepare_data('/opt/airflow/clean_data/fulldata.csv')
    score = compute_model_score(LinearRegression(), X, y)
    context['ti'].xcom_push(key='lr_score', value=score)

def train_dt(**context):
    X, y = prepare_data('/opt/airflow/clean_data/fulldata.csv')
    score = compute_model_score(DecisionTreeRegressor(), X, y)
    context['ti'].xcom_push(key='dt_score', value=score)

def train_rf(**context):
    X, y = prepare_data('/opt/airflow/clean_data/fulldata.csv')
    score = compute_model_score(RandomForestRegressor(), X, y)
    context['ti'].xcom_push(key='rf_score', value=score)

def select_and_save_best_model(**context):
   
    ti = context['ti']
    scores = {
        'lr': ti.xcom_pull(key='lr_score', task_ids='train_lr'),
        'dt': ti.xcom_pull(key='dt_score', task_ids='train_dt'),
        'rf': ti.xcom_pull(key='rf_score', task_ids='train_rf'),
    }

    #  Affichage des scores dans un tableau clair
    print("\n Résumé des scores des modèles :")
    print("{:<20} {:<10}".format("Modèle", "Score (MAE)"))
    print("-" * 30)
    for name, score in scores.items():
        model_full_name = {
            'lr': "Linear Regression",
            'dt': "Decision Tree",
            'rf': "Random Forest"
        }[name]
        print("{:<20} {:.4f}".format(model_full_name, score))

    # Sélection du meilleur modèle

    best_model_name = min(scores, key=lambda k: abs(scores[k]))
    best_score = scores[best_model_name]
    print(f"\n Meilleur modèle sélectionné : {best_model_name.upper()} avec un score de {best_score:.4f}")

    # Entraînement et sauvegarde du modèle
    model_map = {
        'lr': LinearRegression(),
        'dt': DecisionTreeRegressor(),
        'rf': RandomForestRegressor(),
    }

    X, y = prepare_data('/opt/airflow/clean_data/fulldata.csv')
    best_model = model_map[best_model_name]

    save_path = '/opt/airflow/clean_data/best_model.pckl'
    train_and_save_model(best_model, X, y, save_path)

    print(f"\n Modèle sauvegardé avec succès à : {save_path}")

###partie tache
task_get_weather_data = PythonOperator(
    task_id='get_weather_data',
    python_callable=get_weather_data,
    dag=dag,
)

task_transform_last_20 = PythonOperator(
    task_id='transform_last_20_files',
    python_callable=transform_data_into_csv,
    op_kwargs={'n_files': 20, 'filename': 'data.csv'},
    dag=dag,
)

task_transform_all = PythonOperator(
    task_id='transform_all_files',
    python_callable=transform_data_into_csv,
    op_kwargs={'n_files': None, 'filename': 'fulldata.csv'},
    dag=dag,
)
t_train_lr = PythonOperator(
     task_id='train_lr',
     python_callable=train_lr,
     dag=dag)
t_train_dt = PythonOperator(
    task_id='train_dt',
    python_callable=train_dt,
    dag=dag)

t_train_rf = PythonOperator(
        task_id='train_rf',
        python_callable=train_rf,
        dag=dag
)
t_select_model = PythonOperator(
    task_id='select_and_save_model',
    python_callable=select_and_save_best_model,
    dag=dag)
task_get_weather_data >> task_transform_last_20 >> task_transform_all >> [t_train_lr, t_train_dt, t_train_rf]
[t_train_lr, t_train_dt, t_train_rf] >> t_select_model
