
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from src.data_processing import prepare_data

with mlflow.start_run():
    # تحويل النصوص إلى متجهات
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # تدريب النموذج
    model = SVC()
    model.fit(X_train_vec, y_train)

    # تقييم النموذج
    accuracy = model.score(vectorizer.transform(X_test), y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")