FROM python:3.11-slim

WORKDIR /app

# Install dependencies first to cache them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose the port
EXPOSE 8000

CMD ["python", "app.py"]