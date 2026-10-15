# Hugging Face Space (Docker SDK) for the public Clearance demo. Built from a staged copy made by
# scripts/deploy_space.sh, so only the files the demo needs are uploaded.
FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user pyproject.toml ./
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.30" "httpx>=0.27" "mcp>=1.2,<2"
COPY --chown=user src ./src
COPY --chown=user evals ./evals
USER user
ENV PYTHONPATH=/app/src PORT=7860
EXPOSE 7860
# NEBIUS_API_KEY comes from the Space's secrets; without it every run replays Nemotron's saved answers.
CMD ["python", "-m", "clearance", "serve-demo", "--host", "0.0.0.0", "--port", "7860"]
