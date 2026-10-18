# Keycloakによるユーザー管理

Keycloakで登録・認証、PostgreSQLでプレイヤー情報を管理します。
ブラウザはAuthorization Code Flow + PKCE S256でログインし、トークンはメモリ内だけに保持します。
APIはJWKSによるRS256署名、issuer、audience、期限、Bearer種別、発行先クライアントを検証します。

## ローカル起動

```sh
docker compose -f compose.auth.yaml up -d
```

ルートの `.env` に以下を設定します。

```dotenv
NAPOLEON_AUTH_ISSUER=http://localhost:8080/realms/napoleon
NAPOLEON_AUTH_CLIENT_ID=napoleon-web
NAPOLEON_DATABASE_URL=postgresql://napoleon:napoleon-dev@localhost:5433/napoleon
```

```sh
pnpm install --frozen-lockfile
pnpm dev
```

http://localhost:5173/ を開いて新規登録します。Keycloak起動完了前に開いた場合は再読み込みします。
管理画面は http://localhost:8080/admin/ 、開発用管理者は `admin` / `admin-dev` です。
開発設定ではメール認証は無効です。パスワード再設定メールにはSMTP設定が必要です。
既存realmに対して起動時importは上書きされません。設定変更は管理画面等で反映します。

## APIと保存内容

- `GET /api/auth/config`: ブラウザ用の公開認証設定。秘密値は返しません。
- `GET /api/me`: 初回にプレイヤーを作成し、IDとプレイヤー名を返します。
- `PUT /api/me`: `{ "displayName": "名前" }` で1〜30文字のプレイヤー名を保存します。
- Keycloakのissuerとsubの組でプレイヤーを識別します。メール・パスワードはアプリDBに保存しません。
- 認証有効時はhealth/config以外のAPIに認証が必要です。ゲームは作成ユーザーだけが取得・操作できます。
- プレイヤー情報はDB再起動後も残ります。レート対戦は対戦状態・レート・結果をDBに保存します。旧開発用ゲームは従来通りメモリ上です。
- マッチング・CPU対戦・レート更新は [対戦システム](matchmaking.md) を参照してください。
- 認証環境変数を全て省略した場合は従来の開発モードです。一部だけ指定すると起動を拒否します。

## 本番環境

開発用Composeはローカル検証用です。本番では次を設定してください。

- Keycloakを本番の `start` モードでHTTPSの認証ドメインに配置し、issuerをそのrealm URLに合わせる。
- DB・管理者の開発用パスワードを置き換え、DBの永続ボリュームとバックアップを設定する。
- クライアントのValid redirect URIs、Web origins、Valid post logout redirect URIsに実際のアプリURLを登録する。
- 公開クライアント、Standard flow、PKCE S256とAPI用audience mapperを維持する。
- メール認証・パスワード再設定を使う場合はrealmにSMTPを設定する。
- アプリのAPIは同一オリジンの `/api/` または `/napoleon/api/` からFastifyへプロキシする。
- サーバーの3つの認証環境変数を全て設定する。ブラウザにDB接続情報は渡さない。

実環境のドメイン、SMTPとデプロイ先は未設定です。この変更だけで公開環境は作成されません。

## 表示とAPIの権限

Keycloakのrealm role `user` / `admin` を使用します。新規登録には `user` をデフォルト付与します。
管理者はKeycloak管理画面の Users → 対象ユーザー → Role mapping で `admin` を付与してください。
`admin` 単独でも通常プレイ・自分の情報を利用できます。

| 機能 | 未ログイン | user | admin |
| --- | --- | --- | --- |
| 登録・ログイン | ○ | — | — |
| 通常プレイ・自分の情報 | — | ○ | ○ |
| AI設定の変更・AI対戦ログ | — | — | ○ |

一般ユーザーにはAI設定・AI対戦ログのタブを表示しません。APIも独立してロールを検証し、
AIプリセットの詳細取得・更新、シミュレーション、エージェント一覧、ゲーム診断はadminに限定します。
一般ユーザーのCPU選択には、対戦用のCPU一覧APIを利用します。
ロールなしのログインユーザーには利用権限がない旨を表示し、APIは403を返します。
管理者であっても他人のゲームを取得・操作することはできません。

既存realmにはimportが反映されないので、手動でロールを作成し、デフォルトrealm roleに
`user` を追加してください。既存ユーザーにも必要なロールを付与してください。
アクセストークンの `realm_access.roles` にロールが含まれる標準のrolesスコープを維持してください。
変更はトークン更新後に反映されます。認証無効の開発モードでは従来の全画面を利用できます。

本番起動時は `NODE_ENV=production` を設定します。このモードでは認証設定の全省略も起動エラーとなり、認証なしでの起動を防ぎます。
