FROM python:3.12-slim

WORKDIR /usr/src/app

# 必要なパッケージ
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtをコピーしてインストール
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ホストのプロジェクト全体を、コンテナにコピー
COPY . .

# Jupyter Notebook用ポート
EXPOSE 8000

# サーバーを起動
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

