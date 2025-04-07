# Use the NVIDIA CUDA base image
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy the Poetry files
COPY pyproject.toml poetry.lock $PROJECT_ROOT/

RUN poetry config virtualenvs.create false && poetry install --no-cache --no-root;

COPY . $PROJECT_ROOT

RUN PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"