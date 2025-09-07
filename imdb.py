import mlflow
import mlflow.sklearn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
# إعداد الاتصال بـ MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio_user"       
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_pass"   

# -------------------------
# 1. تحميل البيانات
# -------------------------
print("🔹 تحميل بيانات IMDB ...")
dataset = load_dataset("imdb")

# -------------------------
# 2. إعداد عينة صغيرة متوازنة (1000 مثال لكل فئة)
# -------------------------
df = pd.DataFrame({
    "text": dataset["train"]["text"],
    "label": dataset["train"]["label"]
})

df_sample = df.groupby("label").sample(n=1000, random_state=42)

X = df_sample["text"]
y = df_sample["label"]

# -------------------------
# 3. train_test_split مع stratify
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 4. إعداد MLflow
# -------------------------
mlflow.set_tracking_uri("http://localhost:5000")  # عدّل إذا عندك سيرفر خارجي
mlflow.set_experiment("imdb_experiment_windows")

# -------------------------
# 5. تدريب وتجربة
# -------------------------
with mlflow.start_run() as run:
    # Parameters
    max_features = 5000
    C = 5.0

    mlflow.log_param("max_features", max_features)
    mlflow.log_param("C", C)

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model training
    clf = LogisticRegression(max_iter=500, C=C)
    clf.fit(X_train_vec, y_train)

    # Evaluation
    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # -------------------------
    # 6. حفظ Confusion Matrix كـ Artifact
    # -------------------------
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["neg", "pos"],
                yticklabels=["neg", "pos"])
    plt.ylabel("True")
    plt.xlabel("Predicted")

    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/confusion_matrix.png")
    mlflow.log_artifact("artifacts/confusion_matrix.png")

    # -------------------------
    # 7. تسجيل النموذج
    # -------------------------
    mlflow.sklearn.log_model(clf, "model")

    print(f"✅ Run finished with acc={acc:.3f}, f1={f1:.3f}, run_id={run.info.run_id}")
