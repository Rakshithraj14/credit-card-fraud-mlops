FROM python:3.11-slim

WORKDIR /app

# Install only what's needed to serve predictions
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code and model artifacts
COPY app/ ./app/
COPY models/model.pkl models/scaler.pkl ./models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]