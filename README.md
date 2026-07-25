# napoleon-web

ブラウザ向けトランプゲーム「ナポレオン」を開発するための最小構成モノレポです。

今回はナポレオン固有ルールではなく、React画面からActionを送信し、Fastifyサーバー上の純粋なゲームロジックで状態を更新し、ランダムAI 4人が自動でカードを出す縦切りを実装しています。

## 現在の実装範囲

- 5人プレイヤーのゲーム作成
- 52枚デッキから各プレイヤーへ10枚ずつ配布
- 自分の手札表示
- 他プレイヤーは手札枚数のみ表示
- 自分のカードクリックによるAction送信
- サーバー側での状態更新
- ランダムAI 4人の自動プレイ
- 中央の場への5枚表示
- 1トリック完了後の次トリック開始
- 手札がなくなった場合のゲーム終了

## アーキテクチャ概要

- `packages/game-core`: UI、HTTP、AIに依存しないゲーム状態と状態遷移
- `packages/protocol`: フロントエンドとバックエンドで共有するAPI型
- `packages/ai`: AgentインターフェースとランダムAI
- `apps/server`: Fastify API、インメモリゲーム管理、AI自動進行
- `apps/web`: React + ViteのブラウザUI、APIクライアント、表示コンポーネント

Reactコンポーネントにはゲームルールを置かず、合法手はサーバーから返される公開状態に含めています。サーバーは常に人間プレイヤー向けの`createPlayerView`だけを返し、内部状態全体や他プレイヤーの手札内容は公開しません。

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

## テストと検証

```bash
pnpm typecheck
pnpm test
pnpm build
```

## API

- `GET /api/health`
- `POST /api/games`
- `GET /api/games/:gameId`
- `POST /api/games/:gameId/actions`
- `POST /api/games/:gameId/next-trick`

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
- サーバーテストの追加
- Web UIの操作性改善
- WebSocketによるリアルタイム更新
- 永続化、認証、オンライン対戦
