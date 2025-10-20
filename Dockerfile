FROM apache/airflow:2.5.1

USER root

# Install OS packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean

# Upgrade pip system-wide and install dependencies globally (this is safe inside image build)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        opencv-python-headless \
        scikit-learn \
        numpy \
        pandas

# Fix permissions
RUN chown -R airflow:root /opt/airflow

# Switch back to airflow user
USER airflow

