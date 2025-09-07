import argparse
import os
import socket
import platform
import tempfile
import mlflow
import mlflow.sklearn
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import json
import sys

# إعداد الاتصال بـ MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio_user"      
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_pass"  

def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main(args):
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Load data
    data = datasets.load_wine()
    X = data['data']
    y = data['target']
    class_names = data['target_names']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=args.C, max_iter=args.max_iter, solver="lbfgs", multi_class="auto"))
    ])

    with mlflow.start_run(run_name=f"logreg_C{args.C}_seed{args.random_state}") as run:
        # Log params
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Tags
        mlflow.set_tag("host", socket.gethostname())
        mlflow.set_tag("python_version", platform.python_version())
        mlflow.set_tag("sklearn_version", sklearn.__version__)
        mlflow.set_tag("numpy_version", np.__version__)
        mlflow.set_tag("pandas_version", pd.__version__)
        mlflow.set_tag("experiment", "baseline")

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("log_loss", loss)

        # Confusion matrix artifact
        cm = confusion_matrix(y_test, y_pred)
        with tempfile.TemporaryDirectory() as d:
            cm_path = os.path.join(d, "confusion_matrix.png")
            plot_confusion_matrix(cm, class_names, cm_path)
            mlflow.log_artifact(cm_path, artifact_path="plots")

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=args.register_name
        )

        print(json.dumps({
            "run_id": run.info.run_id,
            "metrics": {"accuracy": acc, "log_loss": loss}
        }, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--experiment-name", default="wine-logreg")
    parser.add_argument("--register-name", default="WineClassifier")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args)
