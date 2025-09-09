# train.py
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Tell MLflow to store runs in a local 'mlruns' directory
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Iris Flower Classification DVC")

with mlflow.start_run():
    # --- Load the data from DVC-tracked file ---
    data = pd.read_csv('data/iris.csv')

    # Separate features (X) and target (y)
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and log parameters
    n_estimators = 150 # Changed param to see a new result
    mlflow.log_param("n_estimators", n_estimators)

    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate and log metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    print(f"Model accuracy: {accuracy:.2f}")