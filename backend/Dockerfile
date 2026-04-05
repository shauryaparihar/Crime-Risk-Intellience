FROM python:3.10-slim

WORKDIR /app

# Ensure dependencies are installed first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Pre-train models if they don't exist
RUN python -c "import models; models.train_all()" || echo "Training bypassed or failed; continuing"

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001", "--proxy-headers"]
