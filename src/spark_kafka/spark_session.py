"""
Utility to create Spark sessions configured to connect
to the distributed cluster in Docker.
"""

from pyspark.sql import SparkSession
import socket
import os
from dotenv import load_dotenv

load_dotenv()

def get_spark_session(app_name="Spark Application", extra_conf=None):
    """
    Creates and returns a SparkSession configured to connect to the Spark cluster.
    
    Args:
        app_name (str): Name of the Spark application. Defaults to "Spark Application".
        
    Returns:
        SparkSession: A SparkSession instance connected to the cluster.
        
    Raises:
        Exception: If connection to the Spark cluster fails.
    """
    spark_master_ip = os.getenv("SPARK_MASTER_IP", "localhost")
    master_url = f"spark://{spark_master_ip}:7077"
    driver_host_ip = spark_master_ip
    
    print(f"Attempting to connect to master at: {master_url}")
    print(f"Will announce driver host IP as: {driver_host_ip}")
    
    try:
        socket.gethostbyname(driver_host_ip)
        print(f"Check: IP {driver_host_ip} resolves locally.")
    except socket.gaierror:
        print(f"Warning: IP {driver_host_ip} might not resolve directly on the host machine, this is usually OK.")
    
    try:
        builder = SparkSession.builder \
            .master(master_url) \
            .appName(app_name) \
            .config("spark.driver.host", driver_host_ip)

        if extra_conf:
            for key, value in extra_conf.items():
                print(f"--> Injecting config: {key}={value}")
                builder = builder.config(key, value)

        spark = builder.getOrCreate()
        
        sc = spark.sparkContext
        print("\n--- Connection Successful! --- ✅")
        print(f"Spark version: {sc.version}")
        return spark
        
    except Exception as e:
        print(f"\n--- Connection Failed --- ❌")
        print(f"Error details: {e}")
        raise
