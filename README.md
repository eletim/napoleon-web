# napoleon-web

ブラウザ向けトランプゲーム「ナポレオン」を開発するための最小構成モノレポです。

今回はナポレオン固有ルールではなく、React画面から通信Actionを送信し、Fastifyサーバーが内部Actionへ変換して純粋なゲームロジックで状態を更新し、ランダムAI 4人が自動でカードを出す縦切りを実装しています。

## 現在の実装範囲

- 5人プレイヤーのゲーム作成
- 52枚デッキから各プレイヤーへ10枚ずつ配布
- 自分の手札表示
- 他プレイヤーは手札枚数のみ表示
- 自分のカードクリックによる通信Action送信
- サーバー側での状態更新
- ランダムAI 4人の自動プレイ
- AIが次のリードプレイヤーの場合も、人間の番までサーバー側で自動進行
- 中央の場への5枚表示
- 1トリック完了後のシステム遷移による次トリック開始
- 手札がなくなった場合のゲーム終了

## アーキテクチャ概要

- `packages/game-core`: UI、HTTP、AIに依存しないゲーム状態と状態遷移
- `packages/protocol`: フロントエンドとバックエンドで共有するAPI型
- `packages/ai`: AgentインターフェースとランダムAI
- `apps/server`: Fastify API、インメモリゲーム管理、AI自動進行
- `apps/web`: React + ViteのブラウザUI、APIクライアント、表示コンポーネント

依存方向は、`game-core`を`server`と`ai`が利用し、`protocol`を`server`と`web`が利用する形です。Webは`@napoleon/game-core`を直接importせず、`@napoleon/protocol`の公開DTOだけを共有型として扱います。

Reactコンポーネントにはゲームルールを置かず、合法手はサーバーから返される公開状態に含めています。サーバーは内部の`PlayerView`を`PublicGameState`へ明示変換して返し、内部状態全体や他プレイヤーの手札内容は公開しません。

公開状態は`self`と`opponents`に分かれています。`self`には手札がありますが、`opponents`のDTOには`hand`フィールド自体が存在しません。

ゲーム内部Actionは`playerId`を持つ`play-card`ですが、通信Actionは`cardId`だけを送ります。クライアントは操作対象の`playerId`を送信せず、サーバーがゲームに記録された人間プレイヤーIDを使って内部Actionへ変換します。

次トリックへの進行はプレイヤーActionではありません。`game-core`の`advanceToNextTrick`によるシステム遷移として扱い、現在のリードプレイヤーが人間でもAIでも進められるようにしています。

AI自動進行は一時状態上で実行し、すべて成功した場合だけ正式なゲーム状態へ反映します。AI処理中に失敗した場合、途中まで進んだ状態は保存しません。

## ディレクトリ構成

```text
napoleon-web/
├── apps/
│   ├── web/
│   └── server/
├── packages/
│   ├── game-core/
│   ├── protocol/
│   └── ai/
├── package.json
├── pnpm-workspace.yaml
├── tsconfig.base.json
├── .gitignore
└── README.md
```

## セットアップ

```bash
pnpm install
```

## 開発サーバー

```bash
pnpm dev
```

- Web: http://127.0.0.1:5173
- Server: http://127.0.0.1:3000
- サーバーは`tsx watch`で起動し、サーバーコードとworkspaceパッケージの変更を開発時に反映します。

## テストと検証

```bash
pnpm typecheck
pnpm test
pnpm build
```

`game-core`と`ai`の単体テストに加え、`apps/server`にはFastify `inject`を使ったAPI統合テストがあります。

## API

- `GET /api/health`
- `POST /api/games`
- `GET /api/games/:gameId`
- `POST /api/games/:gameId/actions`
  - リクエスト例: `{ "action": { "type": "play-card", "cardId": "spades-A" } }`
  - `playerId`は送信しません。
- `POST /api/games/:gameId/next-trick`
  - プレイヤーActionではなく、完了済みトリックを次へ進めるシステムAPIです。

## まだ実装していないナポレオン固有ルール

- 競り
- 切り札
- 副官
- フォロー義務
- トリック勝敗判定
- 得点計算
- ジョーカーや特殊カード
- よろめき、セイムツーなどのローカルルール

## 今後の候補

- ナポレオン固有ルールの段階的な追加
- ルール設定の導入
- AI戦略の差し替え
- Web UIの操作性改善
- WebSocketによるリアルタイム更新
- 永続化、認証、オンライン対戦
