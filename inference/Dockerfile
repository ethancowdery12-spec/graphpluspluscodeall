FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY inference_server.py .

# Cloud Run listens on PORT (default 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "inference_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--timeout-keep-alive", "600"]

