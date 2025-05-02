import mlflow
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing

# 1. تحميل البيانات وتنظيفها
df = pd.read_csv('E:\\ready_data.csv')
df = df.dropna(subset=['tweet'])

# 2. ترميز التصنيفات
label_encoder = preprocessing.LabelEncoder()
df['class'] = label_encoder.fit_transform(df["class"])

# 3. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    df['tweet'], 
    df['class'], 
    test_size=0.2, 
    random_state=42
)

# 4. إنشاء مجلدات الحفظ
model_dir = Path("../models/best_model")
model_dir.mkdir(parents=True, exist_ok=True)

with mlflow.start_run(run_name="Arabic_Sentiment_Model") as run:
    # 5. تحويل النصوص إلى متجهات
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    # 6. تدريب النموذج
    model = SVC(C=1.0, kernel='linear', probability=True)
    model.fit(X_train_vec, y_train)

    # 7. تقييم النموذج
    accuracy = model.score(vectorizer.transform(X_test), y_test)
    
    # 8. إعداد مثال الإدخال والتوقيع
    input_example = np.array(X_train.sample(2).values, dtype=str)
    signature = infer_signature(
        input_example,
        model.predict(vectorizer.transform(input_example))
    )

    # 9. تسجيل المعلمات والمقاييس
    mlflow.log_params({
        "model_type": "SVC",
        "vectorizer": "TF-IDF",
        "ngram_range": "(1,2)",
        "max_features": 5000
    })
    
    mlflow.log_metrics({
        "accuracy": accuracy,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })
    
    # 10. حفظ المكونات محلياً
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    # 11. تسجيل النموذج في MLflow مع التوقيع والمثال
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="ArabicSentimentAnalysis"
    )

    print(f"✅ تم حفظ النموذج بنجاح! الدقة: {accuracy:.2f}")
    print(f"🔗 Run ID: {run.info.run_id}")