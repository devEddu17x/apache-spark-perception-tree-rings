FROM apache/spark:4.0.1-java21-python3

USER root

RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    opencv-python-headless \
    numpy \
    pandas \
    scipy \
    boto3 \
    kafka-python==2.0.2 \
    requests \
    python-dotenv \
    tensorflow==2.19.0

RUN mkdir -p /opt/spark/data && chown -R spark:spark /opt/spark/data

USER spark