"""
Script to upload images from local data/raw directory to Cloudflare R2 bucket.
"""

import os
from pathlib import Path
import concurrent.futures
import functools
from r2_config import get_r2_client, BUCKET_NAME

def upload_task(image_file: Path, bucket_prefix: str) -> tuple:
    """
    Task to upload a single image file.
    Designed to be run in a thread pool.
    """
    try:
        s3_client = get_r2_client()
    except Exception as e:
        return (image_file.name, f"Failed to create R2 client: {e}")

    s3_key = f"{bucket_prefix}/{image_file.name}"
    
    try:
        with open(image_file, 'rb') as file_data:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=file_data,
                ContentType='image/png'
            )
        return (image_file.name, "success")
        
    except Exception as e:
        return (image_file.name, str(e))

def upload_images_to_r2(local_dir: str, bucket_prefix: str = "raw"):
    """
    Upload all images from a local directory to Cloudflare R2 bucket
    IN PARALLEL.
    
    Args:
        local_dir (str): Path to local directory containing images.
        bucket_prefix (str): Prefix path in the bucket (default: "raw").
        
    Returns:
        tuple: (successful_uploads, failed_uploads) counts
    """
    # --- Initial check ---
    try:
        # Test R2 client creation to fail fast if credentials are wrong
        get_r2_client()
        print(f"✓ R2 connection test OK. Bucket: {BUCKET_NAME}")
    except Exception as e:
        print(f"✗ Failed to connect to R2: {e}")
        return 0, 0
    
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"✗ Local directory does not exist: {local_dir}")
        return 0, 0
    
    image_files = [f for f in local_path.iterdir() if f.is_file()]
    
    if not image_files:
        print(f"✗ No files found in {local_dir}")
        return 0, 0
    
    print(f"\nFound {len(image_files)} image(s) to upload")
    print("-" * 60)
    
    # Define the number of parallel uploads you want
    MAX_WORKERS = 32
    
    successful_uploads = 0
    failed_uploads = 0
    
    # 'functools.partial' creates a "new" function 'upload_task' 
    # that already has the 'bucket_prefix' argument filled in.
    task_with_prefix = functools.partial(upload_task, bucket_prefix=bucket_prefix)


    print(f"Starting parallel upload with {MAX_WORKERS} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # executor.map executes 'task_with_prefix' for each item in 'image_files'
        # and returns the results in order.
        results = list(executor.map(task_with_prefix, image_files))

    for filename, status in results:
        if status == "success":
            print(f"✓ Uploaded: {filename}")
            successful_uploads += 1
        else:
            print(f"✗ Failed: {filename} -> {status}")
            failed_uploads += 1
    
    # --- Resumen (igual que antes) ---
    print("-" * 60)
    print(f"\nUpload Summary:")
    print(f"  Successful: {successful_uploads}")
    print(f"  Failed: {failed_uploads}")
    print(f"  Total: {len(image_files)}")
    
    return successful_uploads, failed_uploads


# El 'main' no cambia
def main():
    """
    Main function to execute the upload process.
    """
    local_directory = "data/raw"
    bucket_prefix = "raw"
    
    print("=" * 60)
    print("Cloudflare R2 Image Upload Script (Parallel)")
    print("=" * 60)
    print(f"Local directory: {local_directory}")
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Bucket prefix: /{bucket_prefix}")
    print("=" * 60)
    
    successful, failed = upload_images_to_r2(local_directory, bucket_prefix)
    
    if failed > 0:
        print(f"\n✗ Finished with {failed} failed uploads.")
        exit(1)
    else:
        print("\n✓ All uploads completed successfully!")
        exit(0)


if __name__ == "__main__":
    main()