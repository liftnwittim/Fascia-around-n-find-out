FROM python:3.11-slim

ENV CACHE_BUST=20260428

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "FasciaApp.py"]
