"""
Script to upload JSON files from data/ring_annotations directory to Cloudflare R2 bucket
and generate a CSV with the URLs of the uploaded files.
"""

import os
import csv
from pathlib import Path
import concurrent.futures
import functools
from r2_config import get_r2_client, BUCKET_NAME, BASE_DOMAIN


def upload_json_task(json_file: Path, bucket_prefix: str) -> tuple:
    """
    Task to upload a single JSON file.
    Designed to be run in a thread pool.
    
    Returns:
        tuple: (filename, status, url) where status is 'success' or error message
    """
    try:
        s3_client = get_r2_client()
    except Exception as e:
        return (json_file.name, f"Failed to create R2 client: {e}", None)

    s3_key = f"{bucket_prefix}/{json_file.name}"
    
    try:
        with open(json_file, 'rb') as file_data:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=file_data,
                ContentType='application/json'
            )
        
        # Construct the public URL
        url = f"https://{BUCKET_NAME}.{BASE_DOMAIN}/{s3_key}"
        return (json_file.name, "success", url)
        
    except Exception as e:
        return (json_file.name, str(e), None)


def upload_jsons_to_r2(local_dir: str, bucket_prefix: str = "metadata/jsons", 
                       output_csv: str = "metadata/ring_annotations_urls.csv"):
    """
    Upload all JSON files from a local directory to Cloudflare R2 bucket
    IN PARALLEL and generate a CSV with the URLs.
    
    Args:
        local_dir (str): Path to local directory containing JSON files.
        bucket_prefix (str): Prefix path in the bucket (default: "metadata/jsons").
        output_csv (str): Path to output CSV file with URLs.
        
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
    
    # Get all JSON files
    json_files = [f for f in local_path.iterdir() if f.is_file() and f.suffix == '.json']
    
    if not json_files:
        print(f"✗ No JSON files found in {local_dir}")
        return 0, 0
    
    print(f"\nFound {len(json_files)} JSON file(s) to upload")
    print("-" * 60)
    
    # Define the number of parallel uploads
    MAX_WORKERS = 32
    
    successful_uploads = 0
    failed_uploads = 0
    urls_data = []
    
    # Create partial function with bucket_prefix
    task_with_prefix = functools.partial(upload_json_task, bucket_prefix=bucket_prefix)

    print(f"Starting parallel upload with {MAX_WORKERS} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Execute uploads in parallel
        results = list(executor.map(task_with_prefix, json_files))

    # Process results
    for filename, status, url in results:
        if status == "success":
            print(f"✓ Uploaded: {filename}")
            successful_uploads += 1
            urls_data.append({'filename': filename, 'url': url})
        else:
            print(f"✗ Failed: {filename} -> {status}")
            failed_uploads += 1
    
    # --- Generate CSV with URLs ---
    if urls_data:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['filename', 'url']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for row in urls_data:
                    writer.writerow(row)
            
            print(f"\n✓ CSV file generated locally: {output_csv}")
            print(f"  Total URLs: {len(urls_data)}")
            
            # Upload CSV to R2
            try:
                s3_client = get_r2_client()
                csv_key = "metadata/ring_annotations_urls.csv"
                
                with open(output_path, 'rb') as csv_file:
                    s3_client.put_object(
                        Bucket=BUCKET_NAME,
                        Key=csv_key,
                        Body=csv_file,
                        ContentType='text/csv'
                    )
                
                csv_url = f"https://{BUCKET_NAME}.{BASE_DOMAIN}/{csv_key}"
                print(f"✓ CSV uploaded to R2: {csv_url}")
                
            except Exception as e:
                print(f"✗ Failed to upload CSV to R2: {e}")
            
        except Exception as e:
            print(f"\n✗ Failed to generate CSV: {e}")
    
    # --- Summary ---
    print("-" * 60)
    print(f"\nUpload Summary:")
    print(f"  Successful: {successful_uploads}")
    print(f"  Failed:     {failed_uploads}")
    print(f"  Total:      {len(json_files)}")
    
    return successful_uploads, failed_uploads


if __name__ == "__main__":
    # Path to JSON files directory
    json_dir = "data/ring_annotations"
    
    # Upload JSONs to R2 with prefix "metadata/jsons"
    # and generate CSV with URLs
    upload_jsons_to_r2(
        local_dir=json_dir,
        bucket_prefix="metadata/jsons",
        output_csv="metadata/ring_annotations_urls.csv"
    )
