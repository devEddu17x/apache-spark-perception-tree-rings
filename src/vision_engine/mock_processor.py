"""
Mock Image Processor for Spark Streaming Pipeline

This module provides a pure Python function that simulates image processing.
It's designed to be used as a UDF in Spark Structured Streaming.

The function returns a dictionary matching the ResultPayload TypeScript interface:
- jobId: string
- clientId: string (default "unknown")
- status: string ("COMPLETED")
- timestamp: ISO 8601 string
- data: object with processing results
- error: null
"""

import time
import random
from datetime import datetime, timezone


def process_image(job_id: str, file_url: str) -> dict:
    """
    Simulates image processing for a given job.
    
    This is a pure Python function that can be registered as a Spark UDF.
    It simulates processing delay and generates mock results.
    
    Args:
        job_id: Unique identifier for the processing job
        file_url: URL of the image to process
        
    Returns:
        Dictionary matching the ResultPayload interface with:
        - jobId, clientId, status, timestamp
        - data: originalUrl, processedUrl, ringsCount, metadata
        - error: null
    """
    # Simulate processing time
    start_time = time.time()
    time.sleep(0.1)  # 100ms simulated processing
    end_time = time.time()
    
    processing_time_ms = int((end_time - start_time) * 1000)
    
    # Generate mock results
    rings_count = random.randint(20, 30)
    coordinates_x = round(random.uniform(0, 1000), 2)
    coordinates_y = round(random.uniform(0, 1000), 2)
    
    # Get current timestamp in ISO 8601 format
    current_timestamp = datetime.now(timezone.utc).isoformat()
    
    # Build result matching ResultPayload interface
    result = {
        "jobId": job_id,
        "clientId": "unknown",  # Default value as specified
        "status": "COMPLETED",
        "timestamp": current_timestamp,
        "data": {
            "originalUrl": file_url,
            "processedUrl": file_url,  # Same as original for mock
            "ringsCount": rings_count,
            "metadata": {
                "coordinatesX": coordinates_x,
                "coordinatesY": coordinates_y,
                "processingTimeMs": processing_time_ms
            }
        },
        "error": None
    }
    
    return result
