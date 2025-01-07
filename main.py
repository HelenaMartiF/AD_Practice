from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


from functions import params, data_loading, data_preprocess, mlflow_tracking

def main():
    print("Ejecutando main")
    args_values = params()
    df = data_loading()
    x_train, x_test, y_train, y_test = data_preprocess(df)
    
    # Lista de modelos para probar
    modelos = [
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=123)),
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=123)),
        ('DecisionTree', DecisionTreeClassifier(random_state=123)),
        ('SVC', SVC(probability=True, random_state=123))
    ]
    
    # Llama a la función de tracking con los modelos
    mlflow_tracking(args_values.job_name, x_train, x_test, y_train, y_test, modelos)

if __name__ == "__main__":
    main()
