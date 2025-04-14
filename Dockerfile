# Use the pytorch cuda base image
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
ARG PROJECT_ROOT=/app
WORKDIR $PROJECT_ROOT

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ssh git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install -U pip poetry==1.8.3 poetry-core==1.9.0

# Copy the Poetry files first
COPY pyproject.toml poetry.lock $PROJECT_ROOT/

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-cache --no-root

# Copy the rest of the project files
COPY . $PROJECT_ROOT

# Set the PYTHONPATH environment variable
ENV PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

## Expose the port for Jupyter Notebook (if needed)
#EXPOSE 8888
#
## Run Jupyter Notebook or any other command you need
#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--notebook-dir=/app/notebooks"]