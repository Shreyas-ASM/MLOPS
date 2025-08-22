import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from dataset_utils import get_classification_data, get_regression_data
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
experiment_base_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION")
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

if not experiment_base_name:
    raise ValueError("Environment variable MLFLOW_EXPERIMENT_NAME is not set.")

experiment_name = f"train/{experiment_base_name}"
mlflow.set_tracking_uri(tracking_uri)

# Create or get experiment
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
mlflow.set_experiment(experiment_name)

# --------- CLASSIFICATION (SVM) ----------
X_cls, y_cls = get_classification_data()
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="SVM_Classifier"):
    model_cls = SVC(kernel='linear')
    model_cls.fit(X_train_cls, y_train_cls)
    preds = model_cls.predict(X_test_cls)
    acc = accuracy_score(y_test_cls, preds)

    mlflow.log_param("model", "SVM")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model_cls, "svm_model")

    print(f"[SVM] Accuracy: {acc:.4f}")

# --------- REGRESSION (Linear Regression) ----------
X_reg, y_reg = get_regression_data()
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="Linear_Regression"):
    model_reg = LinearRegression()
    model_reg.fit(X_train_reg, y_train_reg)
    preds = model_reg.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, preds)

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model_reg, "linear_regression_model")

    print(f"[Linear Regression] MSE: {mse:.4f}")


