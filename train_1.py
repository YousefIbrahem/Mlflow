import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🔹 ضبط الـ env vars من داخل الكود 
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio_user"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_pass"

# 🔹 تعيين الـ tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris_classification")

# 🔹 تحميل البيانات
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 تدريب الموديل
model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 🔹 تقييم
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# 🔹 توليد signature
signature = infer_signature(X_train, model.predict(X_train))

# 🔹 تسجيل بالـ MLflow
with mlflow.start_run():
    # تسجيل parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)

    # تسجيل metrics
    mlflow.log_metric("accuracy", acc)

    # تسجيل الموديل + signature
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model" , # بدل artifact_path
        registered_model_name="IrisModel",
        signature=signature,
        input_example=X[:5]
    )

    

print("✅ Model, parameters, metrics, and signature logged successfully!")
