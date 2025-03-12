# Base image with CUDA support for ML/LLM workloads
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and essential packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and assign proper permissions
RUN useradd -ms /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Copy only requirements first to leverage caching
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy the config directory into the container
COPY --chown=appuser:appuser config/ ./config

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "src/main.py"]
