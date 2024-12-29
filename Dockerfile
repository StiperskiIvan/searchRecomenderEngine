# Use the official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

# Set the working directory in the container
WORKDIR $APP_HOME

# Copy only the requirements file to install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application and tests into the container
COPY . $APP_HOME

# Run tests before proceeding with the image build
RUN pytest tests/test_search_engine.py

# Expose the port FastAPI will run on
EXPOSE 8000

# Set the default command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
