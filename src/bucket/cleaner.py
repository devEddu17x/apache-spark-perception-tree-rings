import sys

from r2_config import get_r2_client, BUCKET_NAME

# Initialize R2 client
try:
    s3_client = get_r2_client()
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
