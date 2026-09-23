FROM python:3.11-slim

# LightGBM needs OpenMP at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Build listings, local embeddings and the ranker into the image (~40s).
# If you built artifacts yourself (e.g. with Azure embeddings), pass --build-arg BOOTSTRAP=0.
ARG BOOTSTRAP=1
RUN if [ "$BOOTSTRAP" = "1" ]; then python -m scripts.bootstrap; fi

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
