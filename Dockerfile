# REASSEMBLE_VLM：Docker 環境（Linux＋NVIDIA GPU）。用法見 docs/docker.md。
# 映像只放程式、設定、測試與小型索引／切分；原始資料、模型權重、實驗結果、.env（API key）都不放，執行時用 volume 掛進來。
# 版本對齊使用者的 Windows 環境（conda env reassemble-vlm）：Python 3.11、torch 2.14.1（CUDA 12.6 版）。
FROM python:3.11-slim

ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126

ENV PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/workspace/models/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1) 先裝 CUDA 版 torch，再把版本釘住，其他套件就不會把它換掉（和 setup_vlm.bat 同樣做法）
RUN pip install "torch==${TORCH_VERSION}" torchvision --index-url "${TORCH_INDEX}" \
 && python -c "import torch; print('torch==' + torch.__version__)" > /tmp/torch_constraint.txt

# 2) 其他套件（先只複製需求檔：之後只改程式時不用重裝套件）
COPY requirements.txt requirements-vlm.txt requirements-api.txt requirements-train.txt ./
RUN pip install -r requirements.txt -r requirements-vlm.txt -r requirements-api.txt -r requirements-train.txt \
        -c /tmp/torch_constraint.txt

# 3) 程式（以可編輯模式安裝 assembly_monitor，和 setup_windows.bat 相同）
COPY . .
RUN pip install --no-deps -e .

# 預設：跑單元測試（不需要 GPU、資料或模型）
CMD ["python", "-m", "pytest", "-q"]
