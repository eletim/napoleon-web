# 本番配信と運用

構成は Nginx → 静的Web / Fastify / Keycloak、PostgreSQLはアプリ用とKeycloak用の2つです。
外部に公開するポートはNginxの80/443のみ。APIとDBはDocker内部ネットワークで通信します。
アプリサーバーは1プロセスです。全サービスは `restart: unless-stopped` で再起動します。
Docker自体をOS起動時に開始する設定はホスト側で行ってください。

## 初期セットアップ

Docker EngineとCompose v2を用意し、公開DNSをサーバーへ向けます。

```sh
cp .env.production.example .env.production
# DOMAINを実際のホスト名に変更し、3つのpasswordを独立したランダム値に置換
# 例: openssl rand -hex 24
chmod 600 .env.production
node scripts/prepare-production.mjs
```

準備スクリプトは開発用パスワードとサンプルドメインを拒否し、本番ドメインに限定した
Keycloakクライアントのredirect URI / web originを生成します。生成ファイルと.envはGit管理外です。
既存realmは起動時importで上書きされません。開発DBとは別の本番ボリュームを使い、
既存本番realmを更新する際は管理画面から同じredirect URI等を設定してください。

初回の証明書取得時はNginxを停止して80番ポートを空けます。以下のドメインとメールを実値に置き換えます。

```sh
docker compose --env-file .env.production -f compose.production.yaml run --rm -p 80:80 certbot certonly --standalone --non-interactive --agree-tos --email you@example.com -d game.example.com
docker compose --env-file .env.production -f compose.production.yaml up -d --build
```

アクセス先:

- ゲーム: `https://実ドメイン/napoleon/`（`/napoleon`から末尾/へリダイレクト）
- Keycloak管理画面: `https://実ドメイン/auth/admin/`
- API: `/napoleon/api/` をFastifyの `/api/` へ転送

Nginxは静的ビルドを配信し、Vite開発サーバーは使いません。
Keycloakは本番のstartモード、HTTPS公開URL、/authパス、xforwardedを設定します。
Nginxが転送ヘッダーを上書きします。JWKSだけDocker内部URLから取得し、JWTのissuerは公開HTTPS URLで検証します。
Keycloakのuser/adminロールは [ユーザー管理ガイド](user-management.md) の手順で設定します。
SMTPの送信先は環境固有なのでKeycloak管理画面で設定してください。

## 更新・ログ

```sh
docker compose --env-file .env.production -f compose.production.yaml ps
docker compose --env-file .env.production -f compose.production.yaml logs --tail=100 server keycloak nginx
# 更新前のバックアップ
./scripts/database-backup.sh
# コード更新後に再ビルド
node scripts/prepare-production.mjs
docker compose --env-file .env.production -f compose.production.yaml up -d --build
```

SIGTERM/SIGINT時はAPI・タイマー・DB接続を終了します。進行中試合と期限はDBに保存され、再起動後に復元します。
待機列は再募集が必要です。公開設定、証明書、.envは実ホストで管理します。

## 証明書更新

次のコマンドをホストのcron等で定期実行します。スケジュール自体はこの変更では登録しません。

```sh
docker compose --env-file .env.production -f compose.production.yaml run --rm certbot renew --webroot -w /var/www/certbot
docker compose --env-file .env.production -f compose.production.yaml exec -T nginx nginx -s reload
```

## DBバックアップ・復元

```sh
./scripts/database-backup.sh
# または保存先を指定
./scripts/database-backup.sh backups/manual-backup
```

アプリとKeycloakを一時停止して、対応する2つのDBをcustom archive形式で保存し、終了時に再開します。
保存先は権限を絞って作成し、既存dumpには上書きしません。バックアップは別ホストにも保管してください。
`.env.production` と証明書ボリュームはDB dumpには含まれないため、別に保管します。

```sh
# 現在の2つのDBを、指定バックアップの内容で置き換えます。
./scripts/database-restore.sh backups/manual-backup
```

復元時はNginx・アプリ・Keycloakを停止し、アーカイブ検査後に両DBを復元して起動します。
復元に失敗した場合はサービスを停止したままにします。DBコンテナは起動済みである必要があります。
`docker compose down -v` は永続データと証明書を削除するため、通常の更新・停止には使いません。

## この環境での確認範囲

本番用DockerイメージとCompose一式の起動、Nginx実バイナリの構文、自己署名証明書を使ったHTTPS、
実Keycloak・PostgreSQLとブラウザによるE2E、実DBのバックアップ・復元を独立したローカル環境で確認しています。
実公開DNS、Let's Encryptによる証明書の発行・更新、SMTP、ホストOS再起動時の自動起動は未検証です。
