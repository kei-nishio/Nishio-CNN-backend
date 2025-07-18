FROM python:3.12-slim

WORKDIR /usr/src/app

# 必要なパッケージ
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 環境変数設定
ENV PYTHONPATH=/usr/src/app
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes


# requirements.txtをコピーしてインストール
COPY requirements.txt .
RUN pip install --upgrade pip
# 本番環境用（キャッシュを利用しない）
# RUN pip install --no-cache-dir -r requirements.txt
# 開発環境用
RUN pip install -r requirements.txt

# ホストのプロジェクト全体を、コンテナにコピー
# COPY . .

# Jupyter Notebook用ポート
EXPOSE 8888

# Jupyter Notebookを起動
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

## コンテナ起動時に python api_server.py を実行
# CMD ["python", "api_server.py"]

