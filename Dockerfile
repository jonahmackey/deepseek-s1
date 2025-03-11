FROM vllm/vllm-openai
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_SYSTEM_PYTHON=1

COPY . /build
WORKDIR /build
RUN uv sync --group train
