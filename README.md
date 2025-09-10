# Mlflow
mlflow for tracking ml projects
## 📌 Overview
هذا المشروع يوضح كيف يمكن استخدام **MLflow** مع **Docker** لإدارة التجارب في تعلم الآلة، بما يشمل:
- تتبع التجارب (Experiment Tracking).
- تسجيل النماذج (Model Registry).
# dockerfile
python:3.11-slim
mlflow 2.15.1


# Install Dependencies
pip install -r src/requirements.txt


# Run MLflow via Docker
first time to build : docker-compose up -d --build
docker compose up -d   
docker compose down
docker ps -a   # show containers

# Create Virtual Environment
py -3.11 -m venv .venv_mlflow  
pip install mlflow==2.15.1  boto3 scikit-learn pandas psycopg2-binary

# # MLflowتعيين واجهة تتبع 
mlflow.set_tracking_uri("http://localhost:5000")


# MinIOإعداد الاتصال بـ 
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

os.environ["AWS_ACCESS_KEY_ID"] = "minio_user"      

os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_pass"

# minio console 
"http://localhost:9001"





