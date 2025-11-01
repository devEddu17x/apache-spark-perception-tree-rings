import os
import boto3
from dotenv import load_dotenv
import sys

load_dotenv()

# --- Configuration ---
# Replace with your Cloudflare R2 credentials
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
ACCESS_KEY = os.getenv("CLOUDFLARE_ACCESS_KEY")
SECRET_KEY = os.getenv("CLOUDFLARE_SECRET_KEY")
BUCKET_NAME = os.getenv("CLOUDFLARE_BUCKET_NAME")
# ---------------------

# Build the Endpoint URL
ENDPOINT_URL = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"

# 1. Initialize Boto3 client
try:
    s3_client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="auto",  # R2 prefers 'auto' or a valid region
    )
    print(f"Connected to R2. Starting to empty bucket: {BUCKET_NAME}\n")

except Exception as e:
    print(f"Error connecting to R2: {e}")
    sys.exit(1)


# 2. Use a paginator to handle buckets with many objects
paginator = s3_client.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=BUCKET_NAME)

objects_to_delete = []
total_deleted = 0

try:
    for page in pages:
        if "Contents" not in page:
            # No more objects in the bucket
            continue

        # Collect objects from this page
        for obj in page["Contents"]:
            objects_to_delete.append({"Key": obj["Key"]})

        # If we have a batch, delete it (Boto3 handles batches of 1000)
        if objects_to_delete:
            print(f"Deleting a batch of {len(objects_to_delete)} objects...")

            response = s3_client.delete_objects(
                Bucket=BUCKET_NAME, Delete={"Objects": objects_to_delete}
            )

            total_deleted += len(objects_to_delete)

            # Optional: Check for errors in the batch
            if "Errors" in response and response["Errors"]:
                print("Errors occurred while deleting some objects:")
                for error in response["Errors"]:
                    error_msg = error['Message']
                    print(f"  - Key: {error['Key']}, Error: {error_msg}")

            # Clear the list for the next batch
            objects_to_delete = []

    if total_deleted == 0:
        print("The bucket was already empty.")
    else:
        print(f"\nProcess completed. Total objects deleted: {total_deleted}")

except s3_client.exceptions.NoSuchBucket:
    print(f"Error: The bucket '{BUCKET_NAME}' does not exist.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
