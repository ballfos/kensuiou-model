ARG BASE_IMAGE=nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV TZ=Asia/Tokyo
ENV PIP_ROOT_USER_ACTION=ignore

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    python3 \
    python3-pip \
    fonts-noto-cjk \
    wget \
    git \
    && apt-get clean \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# ソースコードのコピー
COPY . .

# 必要なPythonパッケージのインストール
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# デフォルトのコマンド
CMD ["sleep", "infinity"]