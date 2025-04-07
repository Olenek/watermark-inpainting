# Use the pytorch cuda base image
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
ARG PROJECT_ROOT=/app
WORKDIR $PROJECT_ROOT

# Install system dependencies
RUN apt-get update && \
    apt-get clean && \
    apt-get install -y ssh git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install -U pip poetry==1.8.3 poetry-core==1.9.0

# Copy the Poetry files
COPY pyproject.toml poetry.lock $PROJECT_ROOT/

RUN poetry config virtualenvs.create false && poetry install --no-cache --no-root

COPY . $PROJECT_ROOT

RUN PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"