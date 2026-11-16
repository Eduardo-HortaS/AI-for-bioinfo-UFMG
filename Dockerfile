# syntax=docker/dockerfile:1

FROM debian:trixie-20261027-slim

LABEL org.opencontainers.image.title="AI-for-bioinfo-UFMG" \
      org.opencontainers.image.description="Laptop-sized lessons on classical machine learning, neural networks and language models for bioinformatics" \
      org.opencontainers.image.authors="Eduardo-HortaS" \
      org.opencontainers.image.source="https://github.com/Eduardo-HortaS/AI-for-bioinfo-UFMG" \
      org.opencontainers.image.licenses="MIT"

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/

# uv installs the Python each project asks for. It goes in /opt rather than
# root's home so that every user can run it, including the host user that
# Apptainer runs as. Bytecode is compiled now because Apptainer images are
# read-only and Python could not cache it later.
ENV UV_PYTHON_INSTALL_DIR=/opt/lab/python \
    UV_COMPILE_BYTECODE=1

# Build the environments from the project files alone, so that editing a
# lesson does not rebuild them. A single RUN keeps uv's cache and the
# environments in one layer, where identical files (torch, used by two
# modules) are hardlinked instead of stored twice.
COPY build_envs.sh /opt/lab/
COPY --parents jupyter/pyproject.toml jupyter/uv.lock jupyter/overrides.json jupyter/server_config.json \
     modules/*/pyproject.toml modules/*/uv.lock /opt/lab/
RUN UV_CACHE_DIR=/tmp/uv-cache sh /opt/lab/build_envs.sh /opt/lab/envs \
    && rm -rf /tmp/uv-cache

COPY modules /opt/lab/modules
COPY start-lab /usr/local/bin/

RUN useradd --create-home --uid 1000 lab \
    && install -d -o lab -g lab /home/lab/work

USER lab
WORKDIR /home/lab/work

# The environments are read-only for students: `uv run` should use them as
# they are rather than try to update them.
ENV PATH=/opt/lab/envs/jupyter/bin:$PATH \
    UV_NO_SYNC=1 \
    UV_CACHE_DIR=/tmp/uv-cache

EXPOSE 8888
CMD ["start-lab"]
