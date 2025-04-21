# Use Python image as the base image
FROM python:3.10

# Set environment variables 
ENV PYTHONUNBUFFERED 1

# Set working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install EasyOCR
RUN pip install --no-cache-dir easyocr

# Copy all application files into the container
COPY . /app/

# Copy the .env file
COPY .env /app/.env

# Expose the application port
EXPOSE 9000

# Run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "9000"]
