from fastapi import FastAPI
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# تحميل النموذج والمتجهات
model = pickle.load(open("../models/best_model/model.pkl", "rb"))
vectorizer = pickle.load(open("../models/best_model/vectorizer.pkl", "rb"))

@app.post("/predict")
def predict(text: str):
    text_vec = vectorizer.transform([text])
    sentiment = model.predict(text_vec)[0]
    return {"sentiment": sentiment}