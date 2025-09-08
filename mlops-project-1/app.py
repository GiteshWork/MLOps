# app.py

# 1. Import necessary libraries
from flask import Flask, request, jsonify
from joblib import load
import numpy as np

# 2. Create a Flask application instance
# '__name__' is a special variable in Python that gets the name of the current module.
# Flask uses this to know where to look for resources.
app = Flask(__name__)

# 3. Load the trained model
# We load the 'iris_model.joblib' file we created earlier.
# This model object has the 'predict' method we will use.
print("Loading model...")
model = load('iris_model.joblib')
# Let's also save the target names for user-friendly output
iris_target_names = ['Setosa', 'Versicolour', 'Virginica']


# 4. Define an API endpoint for prediction
# '@app.route' is a "decorator" that tells Flask what URL should trigger our function.
# 'methods=['POST']' means this endpoint only accepts POST requests,
# which is standard for sending data to an API.
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives flower measurement data in a POST request,
    uses the loaded model to make a prediction,
    and returns the prediction as a JSON response.
    """
    try:
        # Get the JSON data sent from the client
        data = request.get_json(force=True)

        # The input data is expected to be a list or list of lists,
        # e.g., {'features': [5.1, 3.5, 1.4, 0.2]}
        features = data['features']

        # The model expects a 2D array, so we convert the input.
        # np.array(features).reshape(1, -1) turns [1, 2, 3, 4] into [[1, 2, 3, 4]]
        prediction_input = np.array(features).reshape(1, -1)

        # 5. Use the model to make a prediction
        prediction_id = model.predict(prediction_input)

        # Get the species name from the prediction ID (0, 1, or 2)
        predicted_species = iris_target_names[prediction_id[0]]

        # 6. Return the result as JSON
        # jsonify is a Flask function that converts Python dicts to JSON format.
        return jsonify({
            'prediction': predicted_species,
            'prediction_id': int(prediction_id[0])
        })

    except Exception as e:
        # If anything goes wrong, return an error message
        return jsonify({'error': str(e)}), 400


# 7. Run the Flask application
# 'if __name__ == '__main__':' is a standard Python construct.
# It ensures this code only runs when you execute the script directly.
if __name__ == '__main__':
    # 'host='0.0.0.0'' makes the server accessible from other devices on your network.
    # 'port=5000' is the port it will listen on.
    app.run(host='0.0.0.0', port=5000, debug=True)