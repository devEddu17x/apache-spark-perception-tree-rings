"""
Cloudflare R2 bucket configuration and client initialization.
"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

# R2 Configuration from environment variables
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
ACCESS_KEY = os.getenv("CLOUDFLARE_ACCESS_KEY")
SECRET_KEY = os.getenv("CLOUDFLARE_SECRET_KEY")
BUCKET_NAME = os.getenv("CLOUDFLARE_BUCKET_NAME")
BASE_DOMAIN = os.getenv("BASE_DOMAIN")

# Build the Endpoint URL
ENDPOINT_URL = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"


def get_r2_client():
    """
    Creates and returns a boto3 S3 client configured for Cloudflare R2.
    
    Returns:
        boto3.client: An S3 client configured to connect to Cloudflare R2.
        
    Raises:
        Exception: If connection to R2 fails.
    """
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            region_name="auto",
        )
        return s3_client
    except Exception as e:
        raise Exception(f"Error connecting to R2: {e}")
