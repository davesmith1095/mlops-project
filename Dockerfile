FROM python:3.9-slim

# Organize files into a folder 
WORKDIR /app

# Covers dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy local mlops files to Docker /app
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
