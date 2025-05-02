from fastapi import FastAPI
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# 1. تحديد مسار الملفات
model_path = Path("../models/best_model/model.pkl")
vectorizer_path = Path("../models/best_model/vectorizer.pkl")

# 2. تحميل النموذج والمتجهات مع التحقق من الوجود
try:
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
except FileNotFoundError:
    raise Exception("ملفات النموذج غير موجودة. قم بتدريب النموذج أولاً")

@app.post("/predict")
def predict(text: str):
    # 3. التحقق من أن النص غير فارغ
    if not text.strip():
        return {"error": "النص المدخل فارغ"}
    
    # 4. تحويل النص وتنبؤ المشاعر
    text_vec = vectorizer.transform([text])
    sentiment = model.predict(text_vec)[0]
    
    return {
        "text": text,
        "sentiment": sentiment,
        "model_version": "1.0.0"
    }