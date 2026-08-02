# Build from the repo root so the trained model can be baked in:
#   docker build -f services/serve/Dockerfile -t habari-serve:0.1.0 .
FROM python:3.12-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" transformers sentencepiece \
    protobuf prometheus-fastapi-instrumentator
WORKDIR /app
COPY services/serve/app.py ./
COPY models/afriberta ./models/afriberta
# save_pretrained can write weights with owner-only permissions; the
# container runs as nobody (65534) and must be able to read them
RUN chmod -R a+r ./models/afriberta
ENV MODEL_DIR=/app/models/afriberta HOME=/tmp TORCH_THREADS=2 HF_HUB_OFFLINE=1
USER 65534
EXPOSE 8000
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
