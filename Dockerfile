FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY models/enhanced_attribute_classifier.pth /app/models/
COPY src/ /app/src

# Run the inference service
CMD ["python", "src/inference_service.py"]

