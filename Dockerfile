FROM apache/spark:4.0.1-java21-python3

USER root

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

RUN pip install opencv-python pandas

RUN mkdir -p /opt/spark/data && chown -R spark:spark /opt/spark/data

USER spark