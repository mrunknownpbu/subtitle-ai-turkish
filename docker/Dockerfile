FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    HF_HOME=/models \
    PYTHONPATH=/app/src \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && mkdir -p /models /data /input /output /logs \
    && chmod 777 /models /output /logs

WORKDIR /app
COPY requirements ./requirements
RUN pip install -r requirements/base.txt -r requirements/dev.txt -r requirements/diarize.txt

COPY src ./src
COPY tests ./tests

ENTRYPOINT ["python", "-m", "subai"]
CMD ["--help"]
