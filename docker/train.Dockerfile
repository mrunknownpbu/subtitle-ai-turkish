# Fine-tuning image: the inference image plus QLoRA tooling. Build the inference image first:
#   docker compose -f docker/compose.yml build subai
#   docker build -f docker/train.Dockerfile -t subai-train .
FROM subai:latest
USER root
RUN pip install --no-cache-dir "transformers>=4.44,<4.50" "peft>=0.12,<0.15" "accelerate>=0.33" \
    "bitsandbytes>=0.43" soundfile
ENV HF_HOME=/models
ENTRYPOINT ["python"]
