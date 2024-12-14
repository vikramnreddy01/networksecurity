import subprocess
import sys
from networksecurity.exception.exception import NetworkSecurityException  # Add this import
from networksecurity.logging.logger import logging

class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        command = ["aws", "s3", "sync", folder, aws_bucket_url]
        try:
            subprocess.run(command, check=True)
            print(f"Successfully synced {folder} to {aws_bucket_url}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to sync {folder} to {aws_bucket_url}: {e}")
            raise NetworkSecurityException(f"Failed to sync to S3: {e}", sys)

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        command = ["aws", "s3", "sync", aws_bucket_url, folder]
        try:
            subprocess.run(command, check=True)
            print(f"Successfully synced {aws_bucket_url} to {folder}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to sync from {aws_bucket_url} to {folder}: {e}")
            raise NetworkSecurityException(f"Failed to sync from S3: {e}", sys)
