import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# إعداد الاتصال بـ MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio_user"      
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_pass"  

# إعداد MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-classifier")

def train_and_evaluate(learning_rate, max_iter):
    """ تدريب + تقييم + تسجيل نموذج """
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        # تسجيل المعلمات
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_iter", max_iter)

        # التدريب
        model = LogisticRegression(max_iter=max_iter, C=learning_rate, solver="lbfgs", multi_class="auto")
        model.fit(X_train, y_train)

        # التقييم
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        # Confusion Matrix كـ Artifact
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # تسجيل النموذج
        mlflow.sklearn.log_model(model, "model")

        

def pipeline():
    """ خط كامل لتجارب متعددة """
    experiments = [
        {"learning_rate": 1.0, "max_iter": 200},
        {"learning_rate": 0.5, "max_iter": 300},
        {"learning_rate": 0.1, "max_iter": 500},
    ]

    for exp in experiments:
        train_and_evaluate(exp["learning_rate"], exp["max_iter"])

if __name__ == "__main__":
    pipeline()
