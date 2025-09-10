import os
import mlflow
import boto3
import yaml
from git import Repo

# --- 1. Get Latest Model from MLflow ---
mlflow.set_tracking_uri("mlruns")
runs = mlflow.search_runs(experiment_names=["Iris Flower Classification DVC"])
latest_run_id = runs.iloc[0]['run_id']
model_uri = f"runs:/{latest_run_id}/model"
print(f"Found latest model from run ID: {latest_run_id}")

# --- 2. Upload Model to a Versioned Path in S3 ---
s3_bucket = os.environ['S3_BUCKET_NAME']
s3_model_path = f"models/iris/{latest_run_id}"
s3_full_uri = f"s3://{s3_bucket}/{s3_model_path}"

local_model_path = mlflow.artifacts.download_artifacts(model_uri)
s3_client = boto3.client('s3')
for root, dirs, files in os.walk(local_model_path):
    for file in files:
        local_file_path = os.path.join(root, file)
        s3_client.upload_file(local_file_path, s3_bucket, f"{s3_model_path}/{file}")
print(f"Model successfully uploaded to {s3_full_uri}")

# --- 3. Clone GitOps Repo and Update Manifest ---
gitops_repo_url = os.environ['GITOPS_REPO_URL']
gitops_pat = os.environ['GITOPS_PAT']
cloned_repo_path = "/tmp/gitops_repo"

# Clone the repo using the PAT for authentication
repo_url_with_pat = gitops_repo_url.replace("https://", f"https://oauth2:{gitops_pat}@")
Repo.clone_from(repo_url_with_pat, cloned_repo_path)

manifest_path = os.path.join(cloned_repo_path, "iris-deployment.yaml")
with open(manifest_path, 'r') as f:
    manifest = yaml.safe_load(f)

# Update the storageUri to point to the new, versioned model path
manifest['spec']['predictor']['sklearn']['storageUri'] = s3_full_uri

with open(manifest_path, 'w') as f:
    yaml.dump(manifest, f)
print("Updated iris-deployment.yaml with new model path.")

# --- 4. Commit and Push Changes to GitOps Repo ---
repo = Repo(cloned_repo_path)
repo.config_writer().set_value("user", "name", "github-actions").release()
repo.config_writer().set_value("user", "email", "github-actions@github.com").release()
repo.git.add(update=True)
repo.index.commit(f"Update model to run ID {latest_run_id}")
repo.origin.push()
print("Pushed updated manifest to GitOps repository.")