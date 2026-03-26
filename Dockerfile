FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir ".[train,eval,gradio]"
RUN pip install --no-cache-dir unsloth

COPY configs/ configs/
COPY notebooks/ notebooks/

EXPOSE 7860

ENTRYPOINT ["alignrl"]
CMD ["--help"]
