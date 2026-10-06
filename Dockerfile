# SolarBurst — Production Container
# Multi-stage container with Node.js and Python 3.11 scientific stack

FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NODE_ENV=production \
    PORT=5000 \
    PYTHON_EXEC=python3

# Install system dependencies & Node.js 20
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements and solarburst package
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Install frontend dependencies and build static assets
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# Install backend dependencies
COPY backend/package*.json ./backend/
RUN cd backend && npm install --omit=dev
COPY backend/ ./backend/

# Copy models, configs, and demonstration datasets
COPY models/ ./models/
COPY configs/ ./configs/
COPY data/ ./data/

# Create runtime directories
RUN mkdir -p backend/uploads results

EXPOSE 5000 8000

# Start SolarBurst API and served frontend
CMD ["node", "backend/server.js"]
