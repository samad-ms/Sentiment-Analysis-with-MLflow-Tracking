import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, confusion_matrix
import mlflow

import argparse
import itertools
from typing import List, Tuple


np.random.seed(1234)

def prepare_data(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["sentiment"], random_state=1234) 
    return train_df, test_df

def make_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[csr_matrix, csr_matrix]:
    vectorizer = TfidfVectorizer(stop_words="english")
    train_inputs = vectorizer.fit_transform(train_df["review"])
    test_inputs = vectorizer.transform(test_df["review"])
    return train_inputs, test_inputs

def train(train_inputs, train_outputs: np.ndarray, **model_kwargs) -> BaseEstimator:
    model = LogisticRegression(**model_kwargs)
    model.fit(train_inputs, train_outputs)
    return model

def evaluate(model: BaseEstimator, test_inputs: csr_matrix, test_outputs: np.ndarray, class_names: List[str]) -> Tuple[float, Figure]:
    predicted_test_outputs = model.predict(test_inputs)
    figure = draw_confusion_matrix(test_outputs, predicted_test_outputs, class_names)
    return f1_score(test_outputs, predicted_test_outputs), figure


def draw_confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray, class_names: List[str]) -> Figure:
    labels = list(range(len(class_names)))
    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=labels)

    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Purples)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=20)
    plt.yticks(tick_marks, class_names, fontsize=20)

    fmt = 'd'
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black", fontsize=20)

    plt.title("Confusion matrix")
    plt.ylabel('Actual label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.tight_layout()
    return plt.gcf()


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--penalty", choices=["l1", "l2", "elasticnet", "none"], default="l2")
    parser.add_argument("-C", type=float, default=1.0)
    parser.add_argument("--solver", choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"], default="lbfgs")

    return parser.parse_args()

def main(args):
    df = pd.read_csv("./imdb-dataset.csv")
    df["label"] = pd.factorize(df["sentiment"])[0]

    test_size = 0.3
    train_df, test_df = prepare_data(df, test_size=test_size)

    mlflow.set_experiment("tracking-demo")
    with mlflow.start_run():
        train_inputs, test_inputs = make_features(train_df, test_df)
        model = train(
            train_inputs, 
            train_df["label"].values,
            penalty=args.penalty,
            C=args.C,
            solver=args.solver
        )

        f1_score, figure = evaluate(model, test_inputs, test_df["label"].values, df["sentiment"].unique().tolist())
        figure.savefig("./confusion_matrix.png")
        print("F1 score: ", f1_score)

        mlflow.log_param("test_size", test_size)
        mlflow.log_param("C", args.C)
        mlflow.log_metric("f1_score", f1_score)
        mlflow.log_figure(figure, "figure.png")


if __name__ == "__main__":
    main(parse_args())
