"""
Utility to create Spark sessions configured to connect
to the distributed cluster in Docker.
"""

from pyspark.sql import SparkSession
import socket
import os


def get_spark_session(app_name="Spark Application"):
    """
    Creates and returns a SparkSession configured to connect to the Spark cluster.
    
    Args:
        app_name (str): Name of the Spark application. Defaults to "Spark Application".
        
    Returns:
        SparkSession: A SparkSession instance connected to the cluster.
        
    Raises:
        Exception: If connection to the Spark cluster fails.
    """
    # 1. Master URL (using spark protocol)
    master_url = "spark://localhost:7077"
    
    # 2. Driver Host IP (Laptop's IP on the Docker bridge network)
    # Read from environment variable, fallback to default if not set
    driver_host_ip = os.environ.get("SPARK_MASTER_IP", "172.26.0.1")
    
    print(f"Attempting to connect to master at: {master_url}")
    print(f"Will announce driver host IP as: {driver_host_ip}")
    
    # Optional check (runs on laptop, not inside Docker)
    try:
        socket.gethostbyname(driver_host_ip)
        print(f"Check: IP {driver_host_ip} resolves locally.")
    except socket.gaierror:
        print(f"Warning: IP {driver_host_ip} might not resolve directly on the host machine, this is usually OK.")
    
    # 3. Build SparkSession with driver host configuration
    try:
        spark = SparkSession.builder \
            .master(master_url) \
            .appName(app_name) \
            .config("spark.driver.host", driver_host_ip) \
            .getOrCreate()
        
        # Get the SparkContext
        sc = spark.sparkContext
        
        print("\n--- Connection Successful! --- ✅")
        print(f"SparkSession object: {spark}")
        print(f"SparkContext object: {sc}")
        print(f"Spark version in use: {sc.version}")
        
        return spark
        
    except Exception as e:
        print(f"\n--- Connection Failed --- ❌")
        print(f"Error details: {e}")
        raise
