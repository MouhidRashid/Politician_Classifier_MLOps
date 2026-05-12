# Example: DVC remote setup instructions
# Replace <remote_name> and <remote_url> with your actual remote (e.g., s3, gdrive, azure, etc.)

# Add a DVC remote (example for Google Drive)
dvc remote add -d myremote gdrive://<folder-id>
# Or for S3:
# dvc remote add -d myremote s3://mybucket/path

# Configure credentials if needed
dvc remote modify myremote gdrive_use_service_account true
# dvc remote modify myremote access_key_id <your-access-key>
# dvc remote modify myremote secret_access_key <your-secret-key>

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Check remote status
dvc status -c
