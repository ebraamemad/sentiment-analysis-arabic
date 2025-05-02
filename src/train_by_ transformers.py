import mlflow
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from datasets import Dataset

# 1. تحميل البيانات المحلية باستخدام pandas
data_path = "E:/ready_data.csv"  # حط هنا مسار بياناتك
df = pd.read_csv(data_path)

# 2. تنظيف البيانات: التأكد من أن الأعمدة تحتوي على البيانات الصحيحة (هنا بنفترض أن الأعمدة 'tweet' و 'class')
df = df.dropna(subset=['tweet', 'class'])
df['class'] = df['class'].map({"positive": 0, "negative": 1, "neutral": 2})  # تعديل هذا حسب تصنيفاتك

# 3. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['class'], test_size=0.2, random_state=42)

# 4. تحميل التوكنايزر والنموذج
model_name = "UBC-NLP/MARBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 5. تجهيز البيانات باستخدام التوكنايزر
def tokenize_function(examples):
    return tokenizer(examples['tweet'], padding=True, truncation=True)

train_dataset = Dataset.from_pandas(pd.DataFrame({'tweet': X_train, 'class': y_train}))
train_dataset = train_dataset.map(tokenize_function, batched=True)

test_dataset = Dataset.from_pandas(pd.DataFrame({'tweet': X_test, 'class': y_test}))
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 6. دالة التقييم
def compute_metrics(p):
    predictions, labels = p
    preds = torch.argmax(predictions, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

# 7. إعدادات التدريب
training_args = TrainingArguments(
    output_dir='./results',          # Where to save the model
    evaluation_strategy="epoch",     # تقييم كل مرة بعد كل epoch
    learning_rate=2e-5,              # معدل التعلم
    per_device_train_batch_size=16,  # حجم الدفعة
    per_device_eval_batch_size=16,   # حجم الدفعة أثناء التقييم
    num_train_epochs=3,              # عدد الـ epochs
    weight_decay=0.01,               # لإضافة weight decay
    logging_dir='./logs',            # أين يتم حفظ السجلات
    logging_steps=10,                # كم خطوة بين كل سجل
    save_strategy="epoch",
    load_best_model_at_end=True      # تحميل أفضل موديل عند نهاية التدريب
)

# 8. إنشاء Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 9. بدء التدريب وتسجيل الـ MLflow
with mlflow.start_run(run_name="ArabicSentiment_with_Trainer"):
    # تسجيل المعلمات
    mlflow.log_params(training_args.to_dict())

    # التدريب
    trainer.train()

    # التقييم
    eval_results = trainer.evaluate()

    # تسجيل النتائج
    mlflow.log_metrics({
        "accuracy": eval_results["eval_accuracy"],
        "f1_score": eval_results["eval_f1"]
    })

    # حفظ النموذج محليًا في المسار الذي تم تحديده
    model_dir = "E:/projects/models/best_model"  # المسار الذي تريد حفظ الموديل فيه
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # طباعة النتائج
    print(f"Accuracy: {eval_results['eval_accuracy']:.2f}")
    print(f"F1 Score: {eval_results['eval_f1']:.2f}")

    # تسجيل النموذج في MLflow
    mlflow.pytorch.log_model(model, "model")
