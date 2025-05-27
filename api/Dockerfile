# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.10.17
FROM python:${PYTHON_VERSION}-slim AS base
RUN apt-get update && apt-get install -y \
    build-essential \
    git

# additional dependencies for the application
RUN apt-get install -y \
    poppler-utils \
    tesseract-ocr

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /code

COPY ./api/build.toml ./pyproject.toml
COPY ./api/README.md ./
COPY ./asf_hp_installer_chatbot/config/ ./asf_hp_installer_chatbot/config
COPY ./asf_hp_installer_chatbot/__init__.py ./asf_hp_installer_chatbot/__init__.py
COPY ./rag/ ./rag
COPY ./api/app/ ./app

RUN python -m pip install --no-cache-dir .
RUN python -m pip uninstall -y opencv-python opencv-contrib-python && python -m pip install --no-cache-dir opencv-python-headless==4.10.0.82

# NLTK data things
RUN python -m nltk.downloader punkt -d /usr/local/share/nltk_data
RUN python -m nltk.downloader punkt_tab -d /usr/local/share/nltk_data
RUN python -m nltk.downloader averaged_perceptron_tagger -d /usr/local/share/nltk_data
RUN python -m nltk.downloader averaged_perceptron_tagger_eng -d /usr/local/share/nltk_data

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser
# add permissions to /code
RUN chown -R appuser:appuser /code

# Switch to the non-privileged user to run the application.
USER appuser

EXPOSE 8000

# Specify virtual environment to PATH
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080", "--loop", "uvloop"]
