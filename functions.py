from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import mlflow
import argparse
import subprocess
import time
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

## Params
def params():
    parser = argparse.ArgumentParser(description='__main__')
    parser.add_argument('--job_name', type=str, help='Project name')
    parser.add_argument('--n_estimator_list', nargs='+', type=int, help='List of n_estimators')
    parser.add_argument('--model', type=str, help='Model name')
    return parser.parse_args()	

## Data loading
def data_loading():
    data = load_breast_cancer()
    df = df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    return df

## Data preprocess
def data_preprocess(df):
    train, test = train_test_split(df, test_size=0.2)
    test_target = test['target']
    test[['target']].to_csv('test-target.csv', index=False)
    del test['target']
    test.to_csv('test.csv', index=False)

    features = [x for x in list(train.columns) if x != 'target']
    x_raw = train[features]
    y_raw = train['target']
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw,
                                                        test_size=.20,
                                                        random_state=123,
                                                        stratify=y_raw)
    return x_train, x_test, y_train, y_test

## Model training
def mlflow_tracking(nombre_job, x_train, x_test, y_train, y_test, modelos): 
    print('Ejecutando mlflow_tracking')
    
    mlflow_ui_process = subprocess.Popen(['mlflow', 'ui', '--port', '5000']) 
    print(mlflow_ui_process)
    time.sleep(5)
    
    mlflow.set_experiment(nombre_job) 
    
    for nombre_modelo, modelo_instancia in modelos:
        with mlflow.start_run() as run:
            print(f"Entrenando modelo: {nombre_modelo}")
            
            preprocessor = Pipeline(steps=[('scaler', StandardScaler())])
            model = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', modelo_instancia)])
            
            model.fit(x_train, y_train)
            
            accuracy_train = model.score(x_train, y_train)
            accuracy_test = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

            mlflow.set_tag("mlflow.runName", f"{nombre_modelo}-experiment")
            
            mlflow.log_param('modelo', nombre_modelo)
            mlflow.log_metric('accuracy_train', accuracy_train)
            mlflow.log_metric('accuracy_test', accuracy_test)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('f1_score', f1)
            mlflow.log_metric('roc_auc', roc_auc)
            
            mlflow.sklearn.log_model(model, f'{nombre_modelo}-model')
            
            print(f"Modelo {nombre_modelo} registrado con éxito en MLflow.")
    
    print("Se ha acabado el entrenamiento de los modelos correctamente.")


