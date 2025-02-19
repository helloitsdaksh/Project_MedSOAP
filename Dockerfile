# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables for PyTorch to use GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python as the default command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory inside the container
WORKDIR /workspace

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server with Uvicorn
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]