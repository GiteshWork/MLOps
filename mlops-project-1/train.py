# train.py

# 1. Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# 2. Load the dataset
# The Iris dataset is a classic dataset in machine learning.
# It contains 150 samples of Iris flowers, each with 4 features (sepal length, sepal width, petal length, petal width)
# and the species of the flower.
print("Loading dataset...")
iris = load_iris()
X, y = iris.data, iris.target
# X holds the features (the measurements)
# y holds the labels (the species: 0, 1, or 2)

# 3. Split the data into training and testing sets
# We use some data to train the model and some to test how well it learned.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the model
# A RandomForestClassifier is an effective and easy-to-use model.
# It's like asking many "experts" (decision trees) and taking the majority vote.
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # This is the "learning" step!

# 5. Evaluate the model (optional but good practice)
accuracy = model.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy:.2f}")

# 6. Serialize (save) the trained model to a file
# We save the 'model' object to a file named 'iris_model.joblib'.
print("Saving the model...")
dump(model, 'iris_model.joblib')

print("Model training and saving complete!")