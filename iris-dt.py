import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import  matplotlib.pyplot as plt
import seaborn as sns


import dagshub
dagshub.init(repo_owner='sourav664', repo_name='mlflow-dagshub-demo', mlflow=True)



mlflow.set_tracking_uri("https://dagshub.com/sourav664/mlflow-dagshub-demo.mlflow")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 10
n_estimators = 10

mlflow.set_experiment("iris-dt")

with mlflow.start_run():
    dt = DecisionTreeClassifier(max_depth=max_depth )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    mlflow.log_params({"max_depth": max_depth})
    mlflow.log_metric("accuracy", accuracy)
    # mlflow.log_metric("confusion_matrix", confusion)
    # mlflow.sklearn.log_model(rf, "model")
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    
    mlflow.log_artifact("confusion_matrix.png")
    
    mlflow.log_artifact(__file__)
    
    mlflow.sklearn.log_model(dt, "decision_tree_model")
    
    mlflow.set_tag("author", "hunter")
    mlflow.set_tag('model','decision_tree')