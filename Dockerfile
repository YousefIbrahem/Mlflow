# Dockerfile
FROM python:3.11-slim

# تثبيت الحزم الأساسية أولًا (MLflow + psycopg2 + boto3)
RUN pip install --no-cache-dir --default-timeout=100 \
    mlflow==2.15.1 \
    psycopg2-binary \
    boto3

# تثبيت الحزم الكبيرة بعد ذلك (scikit-learn, pandas)
RUN pip install --no-cache-dir --default-timeout=100 \
    scikit-learn \
    pandas

# تعيين مجلد العمل
WORKDIR /app

# نسخ ملفات المشروع
COPY ./src /app/src

# أمر افتراضي لتشغيل MLflow server
CMD ["mlflow", "server"]
