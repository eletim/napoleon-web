# napoleon-web

ブラウザ向けトランプゲーム「ナポレオン」を開発するための最小構成モノレポです。

React画面から通信Actionを送信し、Fastifyサーバーが内部Actionへ変換して純粋なゲームロジックで状態を更新し、ルールベースAI 4人が自動進行する縦切りを実装しています。

## 現在の実装範囲

- 5人プレイヤーのゲーム作成
- 通常52枚とジョーカー1枚の53枚デッキ
- 各プレイヤーへ10枚ずつ配布し、残り3枚を埋札として扱う
- 自分の手札表示
- 他プレイヤーは手札枚数のみ表示
- 自分を手前、AI4人を左・奥左・奥右・右に配置した5人用テーブルUI
- 中央の場をプレイヤー位置に合わせて表示
- AI4席のコンパクトな統一プレイヤーパネル
- 左右AI席の縦伸びを抑えたテーブル配置
- 自分情報と手札領域を統合した手前側パネル
- 拡大された現在トリック表示
- 控えめな未プレイプレースホルダー
- 操作案内とゲーム状態を分離した上部表示
- 主要ステータスと特殊札情報のコンパクトなチップ表示
- フェーズに応じて必要なステータスだけを表示
- 合法手、不合法手、埋札交換中の選択状態を手札上で表示
- AIの手札は裏向きカードと枚数で表示
- 各プレイヤーが獲得した公開得点札を表示
- 埋札交換結果を一時メッセージとして表示
- ナポレオン、公開済み副官、現在手番をプレイヤー領域に表示
- 契約、公開済み副官、手番メッセージのプレイヤー名を座席名で表示
- PC幅とスマートフォン幅に対応したレスポンシブUI
- スート選択式の競りUI
- `legalActions`に含まれる合法入札候補だけを使った宣言枚数ステッパー
- 現在の最高入札とコンパクトな競り履歴の表示
- 自分のカードクリックによる通信Action送信
- サーバー側での状態更新
- ルールベースAI 4人の自動競り・副官指定・埋札交換・自動プレイ
- ゲーム開始時の競りフェーズ
- 通常入札13〜19
- 競りスート順位: クラブ < ダイヤ < ハート < スペード
- 同じ宣言枚数なら、より強いスートで最高入札を上書き可能
- パス後も、最高入札が更新されれば再び入札可能
- 最新入札者以外の4人連続パスによる契約確定
- 全員パス時の競り開始プレイヤーによるスペード12契約
- 競り結果からナポレオン、切り札、宣言枚数を設定
- 競り終了後の副官指定フェーズ
- 副官札は埋札を見る前に指定
- ナポレオンが通常カードまたはジョーカー1枚を副官札として指定する進行
- 副官指定UIではジョーカーを選択でき、オルマ・よろめき・正ジャック・裏ジャックをショートカットで指定できる
- 指定札が他プレイヤーの手札にある場合、そのプレイヤーを副官として内部解決
- 指定札がナポレオンの手札または埋札にある場合は副官不在として内部解決
- 指定された副官札は公開される
- 副官プレイヤーは指定カードが場に出るまで非公開
- 副官指定後の埋札交換フェーズ
- ナポレオンが埋札3枚を受け取り、13枚の手札から任意の3枚を捨てる進行
- ジョーカーを含む任意のカードを捨てられる交換処理
- 交換後はナポレオンの手札が10枚に戻る
- 捨てた3枚のうち得点札はナポレオンの獲得得点札へ即時加算
- 捨てた3枚のうち非得点札はゲームから除外
- 交換結果イベントでは加算された得点札と非得点札の枚数だけを公開
- 非得点札はゲーム終了後も公開しない
- AIナポレオンの自動副官指定
- AIナポレオンの自動埋札交換
- 交換後はナポレオン先手でプレイ開始
- 標準AIは`RuleBasedAgent`で、競りは手札スコアから契約上限を決め、必要最小枚数だけ競り上げる
- AI副官指定は自分の公開手札にないオルマ・正ジャック・裏ジャック・切り札を評価し、候補がない場合だけジョーカーを含む候補へフォールバックする
- AI埋札交換は13枚から3枚の全組み合わせを評価し、残る10枚のカードスコアを最大化する
- AIプレイはトリック取得確率、得点札期待値、使用カード価値を組み合わせた期待値で合法手を選ぶ
- `RandomAgent`は比較・テスト用に維持
- リードスート
- フォロー義務
- 通常ランクによるトリック勝者判定
- 切り札を考慮したトリック勝者判定
- ジョーカー1枚
- ジョーカーはいつでも出せる
- リードスートを持っていてもジョーカーを出せる
- リードスートがなくてもジョーカーを出す義務はない
- ジョーカー後出しはリードスートの最弱札
- ジョーカー先出しは切り札の最弱札
- ジョーカー先出し時のリードスートは切り札
- 通常札の2はジョーカーより強い
- オルマは常にスペードAで、切り札に依存しない
- オルマは通常、切り札・リードスート・ジョーカーより強い
- よろめき札は常にハートQ
- スペードAとハートQが同じトリックに出た場合、ハートQが勝つ
- スペードAがない場合、ハートQは通常札として扱う
- クラブAはオルマではなく通常札として扱う
- オルマとよろめきは通常スートとしてフォロー義務に従う
- オルマとよろめきはどちらも得点札として扱う
- オルマとよろめきはどちらも埋札へ捨てられる
- オルマとよろめきはどちらも副官札として指定できる
- 正ジャックは切り札スートのJ
- 裏ジャックは切り札と同色のもう一方のJ
- 正ジャック・裏ジャックの対応はスペードとクラブ、ハートとダイヤ
- 正ジャックは裏ジャックより強い
- 正ジャックと裏ジャックは通常の切り札より強い
- オルマは正ジャック・裏ジャックより強い
- よろめき成立時はすべての特殊札より強い
- 正ジャック・裏ジャックは元スートとしてフォロー義務に従う
- 裏ジャックはフォロー義務上の切り札ではない
- 正ジャック・裏ジャックはどちらも得点札として扱う
- 正ジャック・裏ジャックはどちらも埋札へ捨てられる
- 正ジャック・裏ジャックはどちらも副官札として指定できる
- 公開DTOとUIで特殊札ID `orumaCardId` / `yoromekiCardId` / `seiJackCardId` / `uraJackCardId` を表示
- セイムツーは第2トリック以降のみ有効
- セイムツーは5枚すべてが同一スートで、そのスートの2を含む場合に成立
- セイムツー成立時はそのスートの2が勝つ
- オルマ、正ジャック、裏ジャック、ジョーカーを含む場合、セイムツーは不成立
- よろめき札のハートQだけではセイムツーの不成立条件にならない
- セイムツーは勝者判定のみ特殊で、フォロー義務上は通常の2として扱う
- トリック勝者が次のリードプレイヤーになる進行
- 得点札は各スートの10・J・Q・K・A
- 得点札は合計20枚で、各得点札を1枚として数える
- ジョーカーは得点札ではない
- ジョーカーは副官札として指定できる
- ナポレオンと実在する副官が同じ陣営
- 副官不在時はナポレオン単独陣営
- トリック勝者の陣営が、そのトリック内の得点札を獲得
- 埋札交換で捨てた得点札は、ナポレオン本人の獲得得点札として集計済みにする
- 最終得点では埋札得点札を別枠で二重加算しない
- ナポレオン陣営が契約枚数以上なら勝利、未満なら連合軍勝利
- ゲーム終了時に実在する副官を完全公開
- 切り札の公開DTOとUI表示
- ゲームコア上の切り札なし状態
- AIが次のリードプレイヤーの場合も、人間の番までサーバー側で自動進行
- 中央の場への5枚表示
- 1トリック完了後のシステム遷移による次トリック開始
- 手札がなくなった場合のゲーム終了

## アーキテクチャ概要

- `packages/game-core`: UI、HTTP、AIに依存しないゲーム状態と状態遷移
- `packages/protocol`: フロントエンドとバックエンドで共有するAPI型
- `packages/ai`: Agentインターフェース、標準の`RuleBasedAgent`、比較用の`RandomAgent`、再現可能な評価runner
- `apps/server`: Fastify API、インメモリゲーム管理、AI自動進行
- `apps/web`: React + ViteのブラウザUI、APIクライアント、表示コンポーネント

依存方向は、`game-core`を`server`と`ai`が利用し、`protocol`を`server`と`web`が利用する形です。Webは`@napoleon/game-core`を直接importせず、`@napoleon/protocol`の公開DTOだけを共有型として扱います。

Reactコンポーネントにはゲームルールを置かず、合法手はサーバーから返される公開状態に含めています。サーバーは内部の`PlayerView`を`PublicGameState`へ明示変換して返し、内部状態全体や他プレイヤーの手札内容は公開しません。

公開状態は`self`と`opponents`に分かれています。`self`には手札がありますが、`opponents`のDTOには`hand`フィールド自体が存在しません。

ゲーム内部Actionは`playerId`を持ちますが、通信Actionは`playerId`を送りません。クライアントは操作対象の`playerId`を送信せず、サーバーがゲームに記録された人間プレイヤーIDを使って内部Actionへ変換します。

次トリックへの進行はプレイヤーActionではありません。`game-core`の`advanceToNextTrick`によるシステム遷移として扱い、現在のリードプレイヤーが人間でもAIでも進められるようにしています。

AI自動進行は一時状態上で実行し、すべて成功した場合だけ正式なゲーム状態へ反映します。AI処理中に失敗した場合、途中まで進んだ状態は保存しません。

AIにはプレイヤー別の公開ビューを渡します。このビューには現在の切り札、固定の特殊札ID、交換要件、ジョーカー指定可否を含む副官指定要件、公開済みの副官札、埋札交換結果イベントなどの公開情報を含めますが、相手の手札内容、非得点の捨て札内容、未公開の副官プレイヤーIDは含めません。

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
├── start-dev.sh
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
./start-dev.sh
```

または:

```bash
pnpm dev
```

- Web: http://127.0.0.1:5173
- Server: http://127.0.0.1:3000
- サーバーは`tsx watch`で起動し、サーバーコードとworkspaceパッケージの変更を開発時に反映します。

### サーバーAI Policy設定

正式AIはphaseごとのpolicy registryとして`GET /api/agents`の`policyRegistry`に公開されます。server APIでは次のように3軸を独立に構成できます。`nonPlaying`は副官指定と埋札交換を常に1セットとして選択します。

```json
{
  "playing": "ppo-separated-v1000",
  "bidding": "frozen-raise-v1",
  "nonPlaying": "parameterized-adjutant-exchange-v1"
}
```

`parameterized-adjutant-exchange-v1`はrepo-managedの`benchmarks/non-playing-policies/parameterized-adjutant-exchange-v1/policy.json`をhuman-readable source of truthとして直接読みます。schema/version、35+60 weights、SHA、optimizer/verification/dependency provenanceを起動時に検証し、欠損・不正artifactはエラーとして扱います。ONNXへの変換やsilent fallbackは行いません。`frozen-raise-v1`と`ppo-separated-v1000`は既存の正式repo-managed ONNX artifactを使用します。

以下の環境変数ベースのagent設定は後方互換用です。`NAPOLEON_POLICY_1_*`〜`NAPOLEON_POLICY_5_*`はプレイフェーズだけをONNXに差し替えるlegacy playing-only設定です。`DISPLAY_NAME`が空のスロットは無効です。

`NAPOLEON_FULL_POLICY_1_*`〜`NAPOLEON_FULL_POLICY_5_*`もlegacy compatibilityとして残しています。この旧full-policy設定だけがplaying / bidding / adjutant / exchangeすべてのONNXを要求します。新しい正式phase composition pathはこれらの環境変数やfull-ONNX bundleを要求しません。

通常ゲームではphase policyを直接選ばず、server-managedのAI presetを選びます。builtin presetは`COM-RuleBase`（全phase RuleBased）と`COM-AI`（正式採用済み3 policy）の2つです。ゲーム開始時にclientが送るのは`com-rule-base`または`com-ai`のpreset IDだけで、serverが4 AI席すべてに同じphase compositionを解決します。artifact pathやONNX pathはclientへ渡しません。

AI設定画面の保存先はserver process内のpreset storeです。`GET /api/ai-presets`で一覧と利用可能policyを取得し、`PUT /api/ai-presets/:presetId`でcompositionを保存します。ブラウザ再読込後も同じserver process内では保持され、server再起動時は検証済みbuiltin defaultsへ戻ります。初版ではpreset追加・削除やDB永続化は行いません。unknown/unavailable policyは保存時にrejectされ、silent fallbackしません。

```env
NAPOLEON_FULL_POLICY_1_DISPLAY_NAME=Full policy v1
NAPOLEON_FULL_POLICY_1_PLAYING_ONNX_PATH=artifacts/playing/policy.onnx
NAPOLEON_FULL_POLICY_1_PLAYING_METADATA_PATH=artifacts/playing/policy.json
NAPOLEON_FULL_POLICY_1_BIDDING_ONNX_PATH=artifacts/bidding/policy.onnx
NAPOLEON_FULL_POLICY_1_BIDDING_METADATA_PATH=artifacts/bidding/policy.json
NAPOLEON_FULL_POLICY_1_ADJUTANT_ONNX_PATH=artifacts/adjutant/policy.onnx
NAPOLEON_FULL_POLICY_1_ADJUTANT_METADATA_PATH=artifacts/adjutant/policy.json
NAPOLEON_FULL_POLICY_1_EXCHANGE_ONNX_PATH=artifacts/exchange/policy.onnx
NAPOLEON_FULL_POLICY_1_EXCHANGE_METADATA_PATH=artifacts/exchange/policy.json
```

### Tailscale Serve経由のWeb開発アクセス

`./start-dev.sh`と`pnpm dev`は同じ処理です。初回起動時にルートの`.env`がなければ`.env.sample`から生成し、その後`apps/web/.env.local`を確認します。ファイルがなければ`tailscale status --json`の`Self.DNSName`から端末のTailscale DNS名を取得し、末尾の`.`を除去して自動生成します。`VITE_ALLOWED_HOSTS`を読んで、外部アクセス設定がある場合だけTailscale Serveを設定してからVite/Fastifyを起動します。

既存の`.env`や`apps/web/.env.local`は上書きしません。ローカルのONNX PolicyパスはGit管理外の`.env`で調整してください。Tailscale未導入・未接続・DNS名取得失敗などで外部アクセス設定を自動生成できない場合、Tailscale Serveは設定せずlocalhost限定で起動します。設定する場合はGit管理外の`apps/web/.env.local`に端末固有のホスト名だけを書きます。実ホスト名や実IPをリポジトリへ書かないでください。

```env
VITE_ALLOWED_HOSTS=my-machine.example.ts.net
```

この値があると、起動スクリプト内で次のHTTP Serve設定を1回だけ実行します。

```bash
tailscale serve --bg --http=5173 http://127.0.0.1:5173
```

ブラウザでは次の形式でアクセスします。

```text
http://my-machine.example.ts.net:5173/
```

Viteは`127.0.0.1:5173`で待ち受け、Tailscale Serveがtailnet側の5173を転送します。HTTPSやFunnelは使いません。`apps/web/.env.local`に外部アクセス設定がある状態でTailscale未導入・未接続・Serve失敗になった場合は起動を中止します。`start-dev.sh`は自動で`tailscale serve reset`を実行しません。

ブラウザは`/api`へ同一オリジンでアクセスし、Vite開発サーバーが`127.0.0.1:3000`のFastifyへ転送します。Tailscale Serveで公開するのは5173だけで、Fastifyの3000番へ直接アクセスしません。ブラウザコードにlocalhost API URLを設定しないでください。

状態確認と手動解除:

```bash
tailscale serve status
tailscale serve reset
```

### AI対戦ログ

NN学習基盤の最初の単位として、指定seedで5人のルールベースAI対戦を最後まで実行し、判断ログを収集できます。seedは`0`から`4294967295`までのuint32範囲に制限しています。同じseedと同じAgent構成では、配札と行動を再現できるよう、ゲーム用乱数と各Agent用乱数を分離しています。

記録ではAgentへ実際に渡した観測を公開DTOへ変換して保存します。観測には競り状態、競り履歴、特殊札、公開済み副官、埋札交換要求、副官指定要求、現在トリック、完了済みトリック、各プレイヤーの獲得済み得点札を含めます。各判断では合法手の件数だけでなく、Agentへ提示された合法手の実体も保存します。

学習・デバッグ用の完全情報ラベルは観測とは別フィールドです。完全情報には各プレイヤーの手札だけでなく、未使用札、埋札交換で除外された非得点札、埋札処理で獲得扱いになった得点札、現在トリック、完了済みトリックのカードも含め、各判断時点で53枚すべての所在を復元できる形にしています。

ブラウザの`AI対戦ログ`画面では単一対戦の概要、初期カード配置、集計、判断タイムラインを確認できます。判断詳細ではAgentが見ていた観測、合法手一覧、その時点の完全情報ラベルを表示します。

現時点では結果を永続保存しません。必要な場合は画面上のJSONダウンロードで現在表示中の結果を保存できます。

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
  - 競りリクエスト例: `{ "action": { "type": "bid", "suit": "hearts", "targetPointCards": 13 } }`
  - パスリクエスト例: `{ "action": { "type": "pass" } }`
  - 埋札交換リクエスト例: `{ "action": { "type": "discard-cards", "cardIds": ["spades-A", "hearts-2", "joker"] } }`
  - 副官指定リクエスト例: `{ "action": { "type": "choose-adjutant", "cardId": "spades-A" } }`
  - ジョーカー副官指定リクエスト例: `{ "action": { "type": "choose-adjutant", "cardId": "joker" } }`
  - `playerId`は送信しません。
- `POST /api/games/:gameId/next-trick`
  - プレイヤーActionではなく、完了済みトリックを次へ進めるシステムAPIです。

## まだ実装していないナポレオン固有ルール

- ノートランプ契約
- ジョーカー請求
- 得点差に応じたスコア
- プレイヤーごとの精算
- 副官変更
- 高度な陣営AI
- 捨て札制限
- 捨て札による宣言枚数変更
- その他の特殊カード

## 今後の候補

- ナポレオン固有ルールの段階的な追加
- ルール設定の導入
- AI戦略の差し替え
- Web UIの操作性改善
- WebSocketによるリアルタイム更新
- 永続化、認証、オンライン対戦
