"""
Spark Kafka Streaming with Production Vision Pipeline

This module implements a production-grade distributed image processing pipeline using:
- foreachBatch for batch-level control
- mapPartitions for efficient resource management
- Strategy pattern for extensible algorithm design
- Per-algorithm visual outputs with automatic R2 upload

Architecture:
- VisionPipeline is instantiated ONCE per partition (not per row)
- Heavy resources (R2 client, CNN models) initialized once per partition
- Results are flexible JSON structures (not rigid Spark schemas)
- Output is a single String column for maximum flexibility

Key Performance Optimizations:
- No UDF overhead
- One-time resource initialization per partition
- Batch processing with RDD mapPartitions
- Direct Kafka write with checkpoint support
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import json

import sys
import os

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spark_kafka.spark_session import get_spark_session
from spark_kafka.kafka_config import get_kafka_config


# Schema for incoming Kafka messages (matches IngestionPayload interface)
MESSAGE_SCHEMA = StructType([
    StructField("jobId", StringType(), False),
    StructField("file", StringType(), False),
    StructField("timestamp", StringType(), False),
    StructField("clientId", StringType(), False),
    StructField("metadata", StructType([
        StructField("coordinatesX", DoubleType(), False),
        StructField("coordinatesY", DoubleType(), False)
    ]), False)
])


def process_partition(iterator):
    """
    Process a partition of data using VisionPipeline.
    
    CRITICAL: VisionPipeline is initialized ONCE per partition (outside the loop).
    This ensures:
    - R2 client is created once per partition
    - Algorithm models are loaded once per partition
    - Maximum efficiency for distributed processing
    
    This function runs on Spark workers, not on the driver.
    
    Args:
        iterator: Iterator over Row objects in this partition
        
    Yields:
        str: JSON-serialized result for each processed job
    """
    # Import here to ensure it runs on workers
    from vision_engine.core import VisionPipeline
    
    # ✅ CRITICAL: Initialize pipeline ONCE per partition (not per row!)
    pipeline = VisionPipeline()
    
    # Process each row in the partition
    for row in iterator:
        try:
            # Convert Row to dict
            row_dict = row.asDict()
            
            # Process the job
            result = pipeline.process_job(row_dict)
            
            # Serialize to JSON string
            yield json.dumps(result)
            
        except Exception as e:
            # Handle row-level errors
            print(f"❌ Error processing row: {e}")
            import traceback
            traceback.print_exc()
            
            # Yield error result
            error_result = {
                "jobId": row_dict.get('jobId', 'unknown'),
                "clientId": row_dict.get('clientId', 'unknown'),
                "status": "FAILED",
                "timestamp": "",
                "data": {
                    "originalUrl": row_dict.get('file', ''),
                    "metadata": row_dict.get('metadata', {}),
                    "results": {}
                },
                "error": str(e)
            }
            yield json.dumps(error_result)


def process_batch(df, batch_id, spark, kafka_config):
    """
    Process a batch of data using mapPartitions.
    
    This is the foreachBatch handler that:
    1. Converts DataFrame to RDD
    2. Applies mapPartitions for efficient processing
    3. Converts results back to DataFrame
    4. Writes to Kafka
    
    Args:
        df: DataFrame with parsed Kafka messages
        batch_id: Batch identifier
        spark: SparkSession instance
        kafka_config: Kafka configuration object
    """
    if df.isEmpty():
        print(f"Batch {batch_id}: No messages to process")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing Batch {batch_id}")
    print(f"{'='*60}\n")
    
    # Show incoming data
    print("Incoming messages:")
    df.show(truncate=False)
    
    # Convert to RDD and apply mapPartitions
    # This ensures VisionPipeline is initialized once per partition
    results_rdd = df.rdd.mapPartitions(process_partition)
    
    # Check if RDD is empty
    if not results_rdd.isEmpty():
        # Convert RDD back to DataFrame with single String column
        # This is the key to flexible JSON output!
        schema = StructType([
            StructField("value", StringType(), False)
        ])
        
        results_df = spark.createDataFrame(
            results_rdd.map(lambda x: (x,)),  # Wrap string in tuple
            schema
        )
        
        # Write to Kafka
        # IMPORTANT: Don't call .show() before .save() - it triggers duplicate execution!
        results_df.write \
            .format("kafka") \
            .options(**kafka_config.get_output_kafka_options()) \
            .save()
        
        print(f"✅ Batch {batch_id} processed and written to Kafka\n")
    else:
        print(f"⚠️  Batch {batch_id} produced no results\n")


def create_spark_session():
    """Create Spark session with Kafka package."""
    kafka_package = "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0"
    extra_conf = {
        "spark.jars.packages": kafka_package,
        "spark.sql.shuffle.partitions": "4"
    }
    
    spark = get_spark_session("SparkVisionPipeline", extra_conf)
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def create_kafka_stream(spark, kafka_config):
    """
    Create Kafka stream and parse messages.
    
    Args:
        spark: SparkSession instance
        kafka_config: Kafka configuration object
        
    Returns:
        DataFrame: Parsed messages ready for processing
    """
    # Read from Kafka
    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .options(**kafka_config.get_spark_kafka_options()) \
        .load()
    
    # Parse JSON messages
    messages_df = kafka_df.selectExpr("CAST(value AS STRING) as json_string")
    
    parsed_df = messages_df \
        .select(from_json(col("json_string"), MESSAGE_SCHEMA).alias("data")) \
        .select("data.*")
    
    return parsed_df


def main():
    """Main streaming pipeline orchestration."""
    
    print("\n" + "="*60)
    print("Spark Vision Pipeline - Production Architecture")
    print("foreachBatch + mapPartitions + Strategy Pattern")
    print("="*60 + "\n")
    
    # Initialize Spark and Kafka config
    spark = create_spark_session()
    kafka_config = get_kafka_config()
    
    print(f"📥 Reading from topic: {kafka_config.ingestion_topic}")
    print(f"📤 Writing to topic: {kafka_config.results_topic}")
    print(f"🔗 Bootstrap servers: {kafka_config.bootstrap_servers}\n")
    
    try:
        # Create Kafka stream
        stream_df = create_kafka_stream(spark, kafka_config)
        
        print("✅ Kafka stream created successfully")
        print("✅ Using foreachBatch + mapPartitions architecture")
        print("✅ VisionPipeline will initialize once per partition\n")
        
        # Checkpoint location
        checkpoint_location = "/tmp/spark-checkpoint/vision-pipeline"
        
        print(f"💾 Checkpoint location: {checkpoint_location}")
        print("\n🚀 Starting streaming query...\n")
        
        # Use foreachBatch for batch-level control
        # This allows us to use RDD mapPartitions for efficient processing
        query = stream_df \
            .writeStream \
            .foreachBatch(lambda df, batch_id: process_batch(df, batch_id, spark, kafka_config)) \
            .outputMode("append") \
            .option("checkpointLocation", checkpoint_location) \
            .start()
        
        print("✅ Streaming query started successfully!")
        print("⏳ Waiting for messages... (Press Ctrl+C to stop)\n")
        print("="*60 + "\n")
        
        # Wait for termination
        query.awaitTermination()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Keyboard interrupt received")
        print("🛑 Stopping streaming query...")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🧹 Cleaning up...")
        spark.stop()
        print("✅ Spark session stopped")
        print("\n" + "="*60)
        print("Vision Pipeline Terminated")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
