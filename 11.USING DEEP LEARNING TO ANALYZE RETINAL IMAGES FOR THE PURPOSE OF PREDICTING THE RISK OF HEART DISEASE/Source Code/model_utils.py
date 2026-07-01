import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model

DATASET_PATH = "dataset"
RESULT_PATH = "static/results"


# ---------------- LOAD DATA ----------------
def load_test_data():

    X = []
    y = []

    labels = {"normal":0, "risk":1}

    for label in labels:

        folder = os.path.join(DATASET_PATH,label)

        for img in os.listdir(folder):

            path = os.path.join(folder,img)

            image = cv2.imread(path)

            image = cv2.resize(image,(64,64))

            X.append(image)
            y.append(labels[label])

    X = np.array(X)/255.0
    y = np.array(y)

    X_flat = X.reshape(len(X),-1)

    return X, X_flat, y


# ---------------- MODEL METRICS ----------------
def evaluate_models():

    X, X_flat, y_test = load_test_data()

    models = {
        "SVM": joblib.load("models/svm_model.pkl"),
        "Random Forest": joblib.load("models/rf_model.pkl"),
        "KNN": joblib.load("models/knn_model.pkl")
    }

    cnn = load_model("models/cnn_model.h5")

    results = {}
    confusion_data = {}

    for name,model in models.items():

        pred = model.predict(X_flat)

        acc = accuracy_score(y_test,pred)
        precision = precision_score(y_test,pred)
        recall = recall_score(y_test,pred)
        f1 = f1_score(y_test,pred)

        results[name] = {
            "accuracy":acc,
            "precision":precision,
            "recall":recall,
            "f1":f1
        }

        confusion_data[name] = confusion_matrix(y_test,pred)


    # CNN Evaluation
    pred = cnn.predict(X)
    pred = (pred > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)

    results["CNN"] = {
        "accuracy":acc,
        "precision":precision,
        "recall":recall,
        "f1":f1
    }

    confusion_data["CNN"] = confusion_matrix(y_test,pred)

    return results, confusion_data


# ---------------- GENERATE CONFUSION MATRICES ----------------
def generate_confusion_matrices():

    results, confusion_data = evaluate_models()

    for name,cm in confusion_data.items():

        plt.figure(figsize=(4,4))

        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")

        plt.title(name + " Confusion Matrix")

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig(f"{RESULT_PATH}/{name}_cm.png")

        plt.close()


# ---------------- ACCURACY GRAPH ----------------
import matplotlib.pyplot as plt
from model_utils import evaluate_models


def generate_accuracy_graph():

    results, _ = evaluate_models()

    algorithms = []
    accuracies = []

    for name, metrics in results.items():

        algorithms.append(name)
        accuracies.append(metrics["accuracy"] * 100)

    plt.figure(figsize=(7,5))

    plt.bar(algorithms, accuracies, color=["blue","green","orange","red"])

    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy (%)")
    plt.title("Algorithm Accuracy Comparison")

    plt.ylim(0,100)

    plt.savefig("static/results/accuracy_graph.png")

    plt.close()