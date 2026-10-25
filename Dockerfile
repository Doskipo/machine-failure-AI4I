# Slim variant: same Python, without compilers/docs (~150 MB instead of ~1 GB).
FROM python:3.12-slim

# Copy the uv binary straight from its official image instead of pip-installing it.
# Pinned (not :latest) so the same Dockerfile builds the same image next month.
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency files first: this layer and the install below stay cached until they change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Code and model last: they change most often, so only these cheap layers rebuild.
COPY app.py model.joblib ./

# Put the venv on PATH so uvicorn runs directly, without uv needing a writable cache.
ENV PATH="/app/.venv/bin:$PATH"

# Run as an unprivileged user: if the app is compromised, the attacker isn't root in the container.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# 0.0.0.0 so traffic arriving through the -p port mapping is accepted (127.0.0.1 would refuse it).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
