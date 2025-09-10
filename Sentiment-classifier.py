import os
import mlflow
import mlflow.pyfunc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

# إعداد الاتصال بـ MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio_user"       
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_pass"

# تعيين واجهة تتبع MLflow
mlflow.set_tracking_uri("http://localhost:5000")



# إعداد MLflow experiment
mlflow.set_experiment("llm_experiment")

# Parameters
model_name = "distilbert-base-uncased"
num_epochs = 1
batch_size = 8

# تحميل بيانات صغيرة (SST2 - Sentiment Analysis)
dataset = load_dataset("glue", "sst2", split="train[:1%]").train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=64)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# تحميل الموديل
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# دالة حساب الدقة
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# إعداد التدريب
training_args = TrainingArguments(
    output_dir="./results",
    #evaluation_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# تشغيل MLflow Run
with mlflow.start_run(tags={"framework": "transformers", "task": "sentiment-analysis"}):
    # Log Params
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)

    # تدريب
    trainer.train()

    # تقييم
    eval_results = trainer.evaluate()
    mlflow.log_metric("accuracy", eval_results["eval_accuracy"])

    # تسجيل الموديل
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        artifact_path="model"
    )
print("✅ Training finished and logged to MLflow")
