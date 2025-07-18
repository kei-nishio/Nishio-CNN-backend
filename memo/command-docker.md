# 🐳 Docker Compose よく使うコマンド一覧

---

## ✅ 基本操作コマンド

| コマンド                          | 説明                                                         |
| --------------------------------- | ------------------------------------------------------------ |
| `docker compose up`               | コンテナ起動（`docker-compose.yml` に基づく）                |
| `docker compose up -d`            | バックグラウンド（デタッチド）で起動                         |
| `docker compose down`             | コンテナ停止＋ネットワーク・ボリューム削除（イメージは残す） |
| `docker compose build`            | イメージをビルド（Dockerfile 変更反映）                      |
| `docker compose build --no-cache` | キャッシュ無効で再ビルド                                     |
| `docker compose restart`          | コンテナの再起動                                             |
| `docker compose stop`             | コンテナの停止のみ                                           |
| `docker compose start`            | 停止していたコンテナの再開                                   |

---

## 🔍 状態確認・ログ・ステータス

| コマンド                           | 説明                                   |
| ---------------------------------- | -------------------------------------- |
| `docker compose ps`                | 起動中コンテナ一覧（状態確認）         |
| `docker compose logs`              | 全サービスのログ表示                   |
| `docker compose logs -f`           | ログをリアルタイムで追いかける（tail） |
| `docker compose logs [サービス名]` | 特定サービスのログ表示                 |

---

## 🛠️ 操作用コマンド

| コマンド                                      | 説明                                                       |
| --------------------------------------------- | ---------------------------------------------------------- |
| `docker compose exec [サービス名] [コマンド]` | コンテナ内でコマンド実行（例：`bash`）                     |
| `docker compose run [サービス名] [コマンド]`  | 一時的にコンテナを起動してコマンド実行（`exec`とは別扱い） |
| `docker compose config`                       | 設定ファイルをマージ・展開した内容を確認（デバッグ用）     |
| `docker compose rm`                           | 停止中のコンテナを削除（`down`でも削除しきれないときに）   |

---

## 🧹 クリーンアップ・メンテナンス

| コマンド                        | 説明                                        |
| ------------------------------- | ------------------------------------------- |
| `docker compose down -v`        | ボリュームも含めて完全削除（DB なども含む） |
| `docker compose down --rmi all` | イメージも削除（完全初期化に）              |
| `docker compose prune`          | 未使用リソースを削除（v2.22+）              |

---

## 🔄 実務 Tips・オプション

- `up --build`  
  Dockerfile 変更を反映させつつ起動できる便利コマンド。

- `exec -u root`  
  root 権限でコンテナ内に入る（例：`docker compose exec -u root app bash`）。

- `.env` ファイル  
  `docker compose` は `.env` を自動で読み込みます（環境変数定義用）。

---

## 💡 よく使う組み合わせ例（実務向け）

```bash
# 再ビルドしながら起動（デタッチド）
docker compose up -d --build

# アプリケーションコンテナに bash で入る
docker compose exec app bash

# コンテナログをリアルタイムで監視
docker compose logs -f app

# すべて削除（ボリューム含む）
docker compose down -v
```
