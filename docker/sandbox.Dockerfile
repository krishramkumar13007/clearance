# Agent workspace image. No credentials, no route out: the only reachable host is the egress proxy.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*
RUN git config --system --add safe.directory '*' \
    && git config --system user.name "clearance-agent" && git config --system user.email "agent@clearance.local"
WORKDIR /workspace
CMD ["sleep", "infinity"]
