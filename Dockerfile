# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# COPY config.yaml config.yaml
# COPY .env .env
RUN mkdir /documents/
COPY utils/ utils/
COPY my_graph.py my_graph.py 
COPY main.py main.py 
# COPY certs certs

#for normal deploy
# ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.sslCertFile=/app/certs/app.riassume.com+4.pem", "--server.sslKeyFile=/app/certs/app.riassume.com+4-key.pem"]
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

#for Cloud Run deploy
# ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"] 