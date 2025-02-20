import time
import pandas as pd 
import joblib
import os   
import numpy as np

from ladybirdmnist.datasets import LadybirdMNIST
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

if __name__ == "__main__":
    
    dataset_name = "morph-128"
    dpath = "./benchmark/classification/results/machinelearning"
    train_dataset = LadybirdMNIST(root="./data", train=True, download=True, dataset=[dataset_name])
    test_dataset = LadybirdMNIST(root="./data", train=False, download=True, dataset=[dataset_name])

    models = [
        ("SVC", SVC(C=1.0, kernel="rbf", gamma="scale")),
        ("LinearSVC", LinearSVC(C=1.0, max_iter=10000)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        ("Perceptron", Perceptron()),
        ("KNN", KNeighborsClassifier()),
        ("DecisionTree", DecisionTreeClassifier()),
        ("XGBoost", XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
    ]
    
    results = []

    for name, model in models:
        start_time = time.time()
        dsize = np.array(train_dataset.data[0]).shape[1] * np.array(train_dataset.data[0]).shape[2] * np.array(train_dataset.data[0]).shape[3]
        model.fit(np.array(train_dataset.data[0]).reshape(-1, dsize), np.array(train_dataset.label))
        end_time = time.time()
        train_time = end_time - start_time

        start_time = time.time()
        preds = model.predict(np.array(test_dataset.data[0]).reshape(-1, dsize))
        end_time = time.time()
        test_time = end_time - start_time

        acc = accuracy_score(np.array(test_dataset.label), preds)
        results.append((name, acc, train_time, test_time))
        
        model_params = model.get_params()
        model_info = {
            'model': model,
            'parameters': model_params,
            'accuracy': acc,
            'train_time': train_time,
            'test_time': test_time
        }
        os.makedirs(f"{dpath}/{dataset_name}", exist_ok=True)
        df = pd.DataFrame(results, columns=["Model", "Accuracy", "Train Time", "Test Time"])
        df.to_excel(f"./benchmark/results/machinelearning/{dataset_name}/machinelearning_results.xlsx", index=False)
        joblib.dump(model_info, f"./benchmark/results/machinelearning/{dataset_name}/{name}_{acc:.4f}.pkl")

    for name, acc, train_time, test_time in results:
        print(f"{name}: {acc:.4f}, train_time: {train_time:.4f}, test_time: {test_time:.4f}")

    
    
        
