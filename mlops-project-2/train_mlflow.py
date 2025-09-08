# train_mlflow.py

# 1. Import necessary libraries
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("file:./mlruns")
# Set the experiment name in MLflow
mlflow.set_experiment("Iris Flower Classification")

# 2. Start an MLflow run
# 'with mlflow.start_run()' creates a new experiment run.
# MLflow will automatically log the start time, code version, etc.
# All the logging calls inside this 'with' block will be associated with this run.
with mlflow.start_run():
    print("Starting MLflow run...")

    # --- Your original training code ---
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Define model parameters
    # We define params here so we can log them with MLflow.
    n_estimators = 100
    random_state = 42

    # 4. Log parameters to MLflow
    # This records the settings used for this specific run.
    print("Logging parameters...")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions to evaluate the model
    y_pred = model.predict(X_test)

    # 5. Calculate and log metrics
    # This records the performance of our model.
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    print("Logging metrics...")
    mlflow.log_metric("accuracy", accuracy)

    # 6. Log the model itself
    # This saves the model in the MLflow format, making it easy to track and deploy.
    # 'sk_model' is the model object.
    # 'model' is the folder name where the model artifact will be saved.
    print("Logging the model...")
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

    print("\nMLflow run complete!")
    print(f"To see your run, type 'mlflow ui' in your terminal and open the displayed URL.")