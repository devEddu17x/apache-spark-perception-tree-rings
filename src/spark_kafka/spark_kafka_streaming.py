from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark_kafka.kafka_config import get_kafka_config
from notebooks.spark_session import get_spark_session


MESSAGE_SCHEMA = StructType([
    StructField("jobId", StringType(), False),
    StructField("file", StringType(), False),
    StructField("timestamp", StringType(), False),
    StructField("metadata", StringType(), True)
])


class SparkKafkaStreamProcessor:
    def __init__(self, app_name="SparkKafkaImageProcessor"):
        self.spark = self._create_spark_session_with_kafka(app_name)
        self.kafka_config = get_kafka_config()
        self.streaming_query = None
    
    def _create_spark_session_with_kafka(self, app_name):
        spark = get_spark_session(app_name)
        
        spark.sparkContext.setLogLevel("WARN")
        spark.conf.set("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0")
        
        return spark
        
    def create_kafka_stream(self):
        kafka_df = self.spark \
            .readStream \
            .format("kafka") \
            .options(**self.kafka_config.get_spark_kafka_options()) \
            .load()
        
        messages_df = kafka_df.selectExpr("CAST(value AS STRING) as json_string")
        
        parsed_df = messages_df \
            .select(from_json(col("json_string"), MESSAGE_SCHEMA).alias("data")) \
            .select("data.*")
        
        return parsed_df
    
    def process_message(self, batch_df, batch_id):
        if batch_df.isEmpty():
            print(f"Batch {batch_id}: No messages to process")
            return
        
        print(f"\n{'='*60}")
        print(f"Processing Batch {batch_id}")
        print(f"{'='*60}")
        
        batch_df.show(truncate=False)
        
        messages = batch_df.collect()
        
        for msg in messages:
            job_id = msg.jobId
            image_url = msg.file
            timestamp = msg.timestamp
            
            print(f"\nProcessing:")
            print(f"  Job ID: {job_id}")
            print(f"  Image URL: {image_url}")
            print(f"  Timestamp: {timestamp}")
            
            # TODO: Here you will:
            # 1. Download image from R2 using the URL
            # 2. Process the image (create slices)
            # 3. Upload results to R2
            # 4. Send result message to Kafka
            
            print(f"  Status: Ready for processing implementation")
        
        print(f"{'='*60}\n")
    
    def start_streaming(self, processing_mode="foreachBatch"):
        stream_df = self.create_kafka_stream()
        
        if processing_mode == "foreachBatch":
            self.streaming_query = stream_df \
                .writeStream \
                .foreachBatch(self.process_message) \
                .outputMode("append") \
                .start()
                
        elif processing_mode == "console":
            self.streaming_query = stream_df \
                .writeStream \
                .outputMode("append") \
                .format("console") \
                .option("truncate", False) \
                .start()
        
        print(f"\nStreaming started in '{processing_mode}' mode")
        print(f"Listening to topic: {self.kafka_config.ingestion_topic}")
        print(f"Bootstrap servers: {self.kafka_config.bootstrap_servers}")
        print(f"\nWaiting for messages... (Press Ctrl+C to stop)\n")
        
        return self.streaming_query
    
    def await_termination(self):
        if self.streaming_query:
            self.streaming_query.awaitTermination()
    
    def stop(self):
        if self.streaming_query:
            self.streaming_query.stop()
            print("\nStreaming stopped")
        
        if self.spark:
            self.spark.stop()
            print("Spark session stopped")


def main():
    processor = SparkKafkaStreamProcessor()
    
    try:
        query = processor.start_streaming(processing_mode="foreachBatch")
        
        processor.await_termination()
        
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processor.stop()


if __name__ == "__main__":
    main()
