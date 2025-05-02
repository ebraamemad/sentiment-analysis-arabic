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
from mlflow.models import validate_serving_input
import sklearn
import warnings
warnings.filterwarnings('ignore')

# 1. تحميل البيانات وتنظيفها
data_path = Path("E:/ready_data.csv")
df = pd.read_csv(data_path)
df = df.dropna(subset=['tweet', 'class'])

# 2. ترميز التصنيفات
label_encoder = preprocessing.LabelEncoder()
df['class'] = label_encoder.fit_transform(df["class"])

# 3. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    df['tweet'].astype(str),
    df['class'],
    test_size=0.2,
    random_state=42,
    stratify=df['class']
)

# 4. إنشاء مجلدات الحفظ
model_dir = Path("../models/best_model")
model_dir.mkdir(parents=True, exist_ok=True)

# 5. بدء تجربة MLflow
with mlflow.start_run(run_name="Arabic_Sentiment_v2") as run:
    # 6. تحويل النصوص إلى متجهات
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        stop_words=None  # يمكن إضافة كلمات توقف عربية هنا
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # 7. البحث عن أفضل معلمات
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
    
    # 8. التدريب
    grid_search.fit(X_train_vec, y_train)
    best_model = grid_search.best_estimator_
    
    # 9. التقييم
    y_pred = best_model.predict(vectorizer.transform(X_test))
    test_accuracy = best_model.score(vectorizer.transform(X_test), y_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # 10. إعداد مثال الإدخال
    input_example = {"texts": X_train.sample(2).tolist()}
    signature = infer_signature(
        input_example,
        best_model.predict(vectorizer.transform(input_example["texts"]))
    )
    
    # 11. تسجيل كل شيء في MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        "accuracy": test_accuracy,
        "f1_score": f1,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })
    
    # 12. حفظ المكونات محلياً
    artifacts = {
        "model": best_model,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder
    }
    
    for name, obj in artifacts.items():
        with open(model_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    
    # 13. تسجيل النموذج
    mlflow.sklearn.log_model(
        sk_model=artifacts,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="ArabicSentimentAnalysis",
        pip_requirements=[
            f"scikit-learn=={sklearn.__version__}",
            "pandas",
            "mlflow"
        ]
    )
    
    # 14. طباعة النتائج
    print(f"\n✅ تم الانتهاء بنجاح!")
    print(f"🔗 Run ID: {run.info.run_id}")
    print(f"📊 الدقة: {test_accuracy:.2%}")
    print(f"🏆 F1 Score: {f1:.2%}")
    print(f"📝 أفضل معلمات: {grid_search.best_params_}")
    print(f"📂 تم الحفظ في: {model_dir.resolve()}")