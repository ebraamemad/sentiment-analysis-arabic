import mlflow
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from mlflow.models import infer_signature
import sklearn
import warnings

warnings.filterwarnings('ignore')

# ====== تعريف كلاس يجمع كل شيء معاً ======
from sklearn.base import BaseEstimator

class ArabicSentimentModel(BaseEstimator):
    def __init__(self, vectorizer, model, label_encoder):
        self.vectorizer = vectorizer
        self.model = model
        self.label_encoder = label_encoder

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

# ====== تحميل البيانات وتنظيفها ======
data_path = Path("E:/ready_data.csv")
df = pd.read_csv(data_path)
df = df.dropna(subset=['tweet', 'class'])

# ====== ترميز التصنيفات ======
label_encoder = preprocessing.LabelEncoder()
df['class'] = label_encoder.fit_transform(df["class"])

# ====== تقسيم البيانات ======
X_train, X_test, y_train, y_test = train_test_split(
    df['tweet'].astype(str),
    df['class'],
    test_size=0.2,
    random_state=42,
    stratify=df['class']
)

# ====== إنشاء مجلد لحفظ الموديل محليًا ======
model_dir = Path("../models/best_model")
model_dir.mkdir(parents=True, exist_ok=True)

# ====== بدء تجربة MLflow ======
with mlflow.start_run(run_name="Arabic_Sentiment_v2") as run:

    # ====== تحويل النصوص إلى متجهات ======
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        stop_words=None
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    # ====== البحث عن أفضل معلمات SVM ======
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear'],
        'gamma': ['scale']
    }

    grid_search = GridSearchCV(
        SVC(probability=True),
        param_grid,
        cv=3,
        scoring='f1_weighted',
        verbose=1
    )

    # ====== التدريب ======
    grid_search.fit(X_train_vec, y_train)
    best_model = grid_search.best_estimator_

    # ====== التقييم ======
    X_test_vec = vectorizer.transform(X_test)
    y_pred = best_model.predict(X_test_vec)
    test_accuracy = best_model.score(X_test_vec, y_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)

    # ====== إنشاء كائن يحتوي على كل شيء ======
    final_model = ArabicSentimentModel(vectorizer, best_model, label_encoder)

    # ====== إعداد signature و input example ======
    input_texts = X_train.sample(2).tolist()
    input_example = {"texts": input_texts}
    signature = infer_signature(
        pd.DataFrame({"texts": input_texts}),
        final_model.predict(input_texts)
    )

    # ====== تسجيل المعلمات والنتائج ======
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        "accuracy": test_accuracy,
        "f1_score": f1,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })

    # ====== تسجيل النموذج في MLflow ======
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        signature=signature,
        input_example=pd.DataFrame({"texts": input_texts}),
        registered_model_name="ArabicSentimentAnalysis",
       
    )

    # ====== حفظ الملفات محليًا كنسخة احتياطية ======
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    # ====== طباعة النتائج ======
    print(f"\n✅ تم الانتهاء بنجاح!")
    print(f"🔗 Run ID: {run.info.run_id}")
    print(f"📊 الدقة: {test_accuracy:.2%}")
    print(f"🏆 F1 Score: {f1:.2%}")
    print(f"📝 أفضل معلمات: {grid_search.best_params_}")
    print(f"📂 تم الحفظ في: {model_dir.resolve()}")
