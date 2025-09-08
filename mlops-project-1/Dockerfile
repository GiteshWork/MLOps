# Step 1: Start from an official Python base image.
# We use python:3.9-slim as it's a smaller, more secure base.
FROM python:3.9-slim

# Step 2: Set the working directory inside the container.
# All subsequent commands will run from here.
WORKDIR /app

# Step 3: Copy the requirements file into the container.
COPY requirements.txt .

# Step 4: Install the Python dependencies.
# --no-cache-dir makes the image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application code into the container.
# This includes app.py and our trained model iris_model.joblib.
COPY . .

# Step 6: Tell Docker what command to run when the container starts.
# This will execute 'python app.py' to start our Flask server.
CMD ["python", "app.py"]