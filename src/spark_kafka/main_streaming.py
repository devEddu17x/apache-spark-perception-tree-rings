"""
Spark Kafka Streaming with UDF-based Processing

This module implements a distributed Spark Structured Streaming pipeline that:
1. Reads messages from Kafka (image processing jobs)
2. Applies a UDF to process each message in the workers (distributed)
3. Writes results back to Kafka (no collect() anti-pattern)

Architecture:
- Uses UDFs for distributed processing (runs on workers, not driver)
- Avoids foreachBatch + collect() anti-pattern
- Writes directly to Kafka using writeStream
- Maintains checkpoints for fault tolerance
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct, udf
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, LongType
)

import sys
import os

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spark_kafka.spark_session import get_spark_session
from spark_kafka.kafka_config import get_kafka_config
from vision_engine.mock_processor import process_image


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


# Schema for UDF return value (matches ResultPayload TypeScript interface)
RESULT_SCHEMA = StructType([
    StructField("jobId", StringType(), False),
    StructField("clientId", StringType(), False),
    StructField("status", StringType(), False),
    StructField("timestamp", StringType(), False),
    StructField("data", StructType([
        StructField("originalUrl", StringType(), False),
        StructField("processedUrl", StringType(), False),
        StructField("ringsCount", IntegerType(), False),
        StructField("metadata", StructType([
            StructField("coordinatesX", DoubleType(), False),
            StructField("coordinatesY", DoubleType(), False),
            StructField("processingTimeMs", IntegerType(), False)
        ]), False)
    ]), False),
    StructField("error", StringType(), True)
])


def create_spark_session():
    """Create Spark session with Kafka package."""
    kafka_package = "org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.0"
    extra_conf = {
        "spark.jars.packages": kafka_package,
        "spark.sql.shuffle.partitions": "4"
    }
    
    spark = get_spark_session("SparkKafkaUDFProcessor", extra_conf)
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def main():
    """Main streaming pipeline orchestration."""
    
    print("\n" + "="*60)
    print("Spark Kafka Streaming with UDF-based Processing")
    print("="*60 + "\n")
    
    # Initialize Spark and Kafka config
    spark = create_spark_session()
    kafka_config = get_kafka_config()
    
    print(f"📥 Reading from topic: {kafka_config.ingestion_topic}")
    print(f"📤 Writing to topic: {kafka_config.results_topic}")
    print(f"🔗 Bootstrap servers: {kafka_config.bootstrap_servers}\n")
    
    try:
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
        
        print("✅ Kafka stream created successfully")
        
        # Register UDF for distributed processing
        # This runs on workers, not on the driver!
        process_udf = udf(process_image, RESULT_SCHEMA)
        
        print("✅ UDF registered: process_image")
        
        # Apply UDF to process each message
        # This happens in a distributed manner across workers
        # Pass clientId and metadata to preserve original data
        processed_df = parsed_df.withColumn(
            "result",
            process_udf(
                col("jobId"), 
                col("file"),
                col("clientId"),
                col("metadata")
            )
        )
        
        # Select only the result column and flatten it
        result_df = processed_df.select("result.*")
        
        # Convert result to JSON for Kafka value field
        output_df = result_df.select(
            to_json(struct("*")).alias("value")
        )
        
        print("✅ Processing pipeline configured")
        
        # Write to Kafka (no foreachBatch, no collect!)
        # This is the correct way to use Spark Structured Streaming
        checkpoint_location = "/tmp/spark-checkpoint/kafka-udf-processor"
        
        print(f"💾 Checkpoint location: {checkpoint_location}")
        print("\n🚀 Starting streaming query...\n")
        
        query = output_df \
            .writeStream \
            .format("kafka") \
            .options(**kafka_config.get_output_kafka_options()) \
            .option("checkpointLocation", checkpoint_location) \
            .outputMode("append") \
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
        print("Streaming pipeline terminated")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
