import os
from dotenv import load_dotenv

load_dotenv()


class KafkaConfig:
    def __init__(self):
        self.bootstrap_servers = os.getenv(
            'KAFKA_BOOTSTRAP_SERVERS', 
            'localhost:29092,localhost:39092,localhost:49092'
        )
        self.ingestion_topic = os.getenv('KAFKA_INGESTION_TOPIC', 'image-ingestion')
        self.results_topic = os.getenv('KAFKA_RESULTS_TOPIC', 'processing-results')
        self.group_id = os.getenv('KAFKA_GROUP_ID', 'spark-image-processor')
        
    def get_spark_kafka_options(self):
        return {
            'kafka.bootstrap.servers': self.bootstrap_servers,
            'subscribe': self.ingestion_topic,
            'startingOffsets': 'latest',
            'failOnDataLoss': 'false',
            'kafka.group.id': self.group_id
        }
    
    def get_output_kafka_options(self):
        return {
            'kafka.bootstrap.servers': self.bootstrap_servers,
            'topic': self.results_topic
        }


def get_kafka_config():
    return KafkaConfig()
