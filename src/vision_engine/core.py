"""
Vision Engine Core Module

This module provides the core infrastructure for the distributed image processing pipeline:
- BaseVisionAlgorithm: Abstract base class for all vision algorithms
- VisionPipeline: Orchestrator that manages R2 connections, downloads images, 
  executes algorithms, and handles visual outputs

Architecture:
- Uses Strategy pattern for extensible algorithm design
- Initializes heavy resources (R2 client, models) once per partition
- Supports per-algorithm visual outputs via _visual_output convention
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional
import traceback
import io

import cv2
import numpy as np
import requests

# Import R2 configuration
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spark_kafka.r2_config import get_r2_client, BUCKET_NAME, BASE_DOMAIN


class BaseVisionAlgorithm(ABC):
    """
    Abstract base class for all vision processing algorithms.
    
    IMPORTANT: Subclasses should load heavy resources (CNN models, TensorFlow/PyTorch
    models, etc.) in their __init__ method. This ensures one-time initialization 
    per partition when using mapPartitions in Spark.
    
    Example:
        class MyDetector(BaseVisionAlgorithm):
            def __init__(self):
                # Load model ONCE per partition
                self.model = tf.keras.models.load_model('my_model.h5')
            
            def process(self, image_numpy, metadata):
                predictions = self.model.predict(image_numpy)
                return {"detections": predictions.tolist()}
    """
    
    @abstractmethod
    def process(self, image_numpy: np.ndarray, metadata: dict) -> dict:
        """
        Process an image and return results.
        
        Args:
            image_numpy: OpenCV image as numpy array (BGR format)
            metadata: Original metadata from ingestion payload
            
        Returns:
            dict: Algorithm-specific results. Can include:
                - Any data structure (nested dicts, lists, primitives)
                - Optional '_visual_output' key with numpy array for processed image
                  (will be automatically uploaded to R2 and replaced with 'imageUrl')
        
        Example return with visual output:
            {
                "count": 25,
                "details": {"confidence": 0.9},
                "_visual_output": processed_image_numpy  # Optional
            }
        """
        pass


class VisionPipeline:
    """
    Orchestrator for the vision processing pipeline.
    
    Responsibilities:
    - Initialize R2 client (once per partition)
    - Instantiate and manage vision algorithms
    - Download images from R2/URLs
    - Execute all algorithms on each image
    - Handle per-algorithm visual outputs (_visual_output convention)
    - Upload processed images to R2
    - Accumulate results and handle errors gracefully
    
    This class is designed to be instantiated ONCE per partition in mapPartitions,
    ensuring efficient resource usage.
    """
    
    def __init__(self):
        """
        Initialize the vision pipeline.
        
        CRITICAL: This is called ONCE per partition, not per row.
        Heavy initialization (R2 client, algorithm models) happens here.
        """
        print("🔧 Initializing VisionPipeline (once per partition)...")
        
        # Initialize R2 client
        self.r2_client = get_r2_client()
        self.bucket_name = BUCKET_NAME
        self.base_domain = BASE_DOMAIN
        
        # Import and instantiate algorithms
        # This happens once per partition, not per image
        from vision_engine.algorithms import RingDetection, UnsharpMasking
        
        self.algorithms = [
            RingDetection(),
            UnsharpMasking()
        ]
        
        print(f"✅ VisionPipeline initialized with {len(self.algorithms)} algorithms")
    
    def _download_image(self, url: str) -> Optional[np.ndarray]:
        """
        Download image from URL and decode to numpy array.
        
        Args:
            url: Image URL (can be R2 URL or any HTTP URL)
            
        Returns:
            numpy.ndarray: Decoded image in BGR format (OpenCV), or None if failed
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Decode image from bytes
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"❌ Failed to decode image from {url}")
                return None
            
            print(f"✅ Downloaded image: {image.shape}")
            return image
            
        except Exception as e:
            print(f"❌ Error downloading image from {url}: {e}")
            return None
    
    def _upload_result(self, image_numpy: np.ndarray, filename: str) -> str:
        """
        Upload processed image to R2 and return public URL.
        
        Args:
            image_numpy: Processed image as numpy array
            filename: Filename to use in R2 (e.g., "job123_ring_detection.png")
            
        Returns:
            str: Public URL of uploaded image
        """
        try:
            # Encode image to PNG
            success, buffer = cv2.imencode('.png', image_numpy)
            if not success:
                raise Exception("Failed to encode image to PNG")
            
            # Upload to R2
            key = f"app/results/{filename}"
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.tobytes(),
                ContentType='image/png'
            )
            
            # Construct public URL
            public_url = f"https://{self.base_domain}/{key}"
            print(f"✅ Uploaded image to: {public_url}")
            
            return public_url
            
        except Exception as e:
            print(f"❌ Error uploading image {filename}: {e}")
            # Return a placeholder URL on error
            return f"https://{self.base_domain}/error/upload_failed.png"
    
    def process_job(self, row_dict: dict) -> dict:
        """
        Process a single job (image processing request).
        
        This method:
        1. Downloads the image once
        2. Runs all algorithms on the image
        3. Detects _visual_output in algorithm results
        4. Uploads visual outputs to R2 and replaces with imageUrl
        5. Accumulates all results
        6. Returns structured JSON-serializable dict
        
        Args:
            row_dict: Dictionary from Kafka message with keys:
                - jobId, clientId, file (URL), timestamp, metadata
        
        Returns:
            dict: Complete result payload ready for JSON serialization
        """
        job_id = row_dict.get('jobId', 'unknown')
        client_id = row_dict.get('clientId', 'unknown')
        file_url = row_dict.get('file', '')
        timestamp = row_dict.get('timestamp', '')
        metadata = row_dict.get('metadata', {})
        
        print(f"\n{'='*60}")
        print(f"Processing Job: {job_id}")
        print(f"{'='*60}")
        
        try:
            # Download image once
            image = self._download_image(file_url)
            
            if image is None:
                raise Exception(f"Failed to download image from {file_url}")
            
            # Execute all algorithms and accumulate results
            results = {}
            
            for algo in self.algorithms:
                algo_name = algo.__class__.__name__
                
                # Convert to snake_case for result key
                algo_key = ''.join(['_' + c.lower() if c.isupper() else c 
                                   for c in algo_name]).lstrip('_')
                
                print(f"  🔄 Running {algo_name}...")
                
                try:
                    # Execute algorithm
                    algo_result = algo.process(image, metadata)
                    
                    # Check for visual output
                    if '_visual_output' in algo_result:
                        visual_output = algo_result.pop('_visual_output')
                        
                        # Upload to R2
                        filename = f"{job_id}_{algo_key}.png"
                        image_url = self._upload_result(visual_output, filename)
                        
                        # Add imageUrl to result
                        algo_result['imageUrl'] = image_url
                    
                    # Add to results
                    results[algo_key] = algo_result
                    print(f"  ✅ {algo_name} completed")
                    
                except Exception as algo_error:
                    print(f"  ❌ {algo_name} failed: {algo_error}")
                    results[algo_key] = {
                        "error": str(algo_error),
                        "status": "failed"
                    }
            
            # Build final result payload
            result_payload = {
                "jobId": job_id,
                "clientId": client_id,
                "status": "COMPLETED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "originalUrl": file_url,
                    "metadata": metadata,
                    "results": results
                },
                "error": None
            }
            
            print(f"✅ Job {job_id} completed successfully")
            return result_payload
            
        except Exception as e:
            # Handle job-level errors
            print(f"❌ Job {job_id} failed: {e}")
            traceback.print_exc()
            
            return {
                "jobId": job_id,
                "clientId": client_id,
                "status": "FAILED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {
                    "originalUrl": file_url,
                    "metadata": metadata,
                    "results": {}
                },
                "error": str(e)
            }
