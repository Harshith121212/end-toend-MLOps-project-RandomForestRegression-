# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only the necessary files
COPY score.py .

# Install only Flask (no sklearn, joblib, etc. since we aren’t using model yet)
RUN pip install --no-cache-dir flask

# Expose port
EXPOSE 5000

# Run Flask app
CMD ["python", "score.py"]
