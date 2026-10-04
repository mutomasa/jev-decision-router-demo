# CLAUDE.md

Jev（System 1 / 判断レイヤー）とローカル vLLM（System 2 / 生成レイヤー）を組み合わせた Streamlit デモ。
概要・セットアップは README.md を参照。

## コマンド

パッケージ管理は **uv** を使う。`pip` や `requirements.txt` は使わない。

```bash
uv sync                                         # 依存関係のインストール（dev グループ含む）
uv add <pkg> / uv add --dev <pkg>               # 依存関係の追加
uv run pytest                                   # テスト
uv run ruff check . && uv run ruff format --check .  # lint / フォーマット確認
uv run ruff check --fix . && uv run ruff format .    # 自動修正
uv run mypy                                     # 型チェック（strict）
uv run pre-commit install                       # コミット時に ruff / mypy を自動実行
uv run streamlit run src/jev_decision_router/app.py  # アプリ起動
uv run jev-router-eval [--mock]                 # ルーティング精度の評価
```

## 完了の定義

変更を終える前に、必ず以下をすべて通すこと。

1. `uv run ruff check .`
2. `uv run ruff format --check .`
3. `uv run mypy`
4. `uv run pytest`（カバレッジ 90% 未満で失敗する）

同じチェックを GitHub Actions（`.github/workflows/ci.yml`）でも実行している。

**pytest は必須。** 新しい機能・バグ修正には必ずテストを追加する。外部 API（Jev / vLLM）を実際に呼ぶテストは書かない。
HTTP クライアントは `httpx.MockTransport`（`tests/conftest.py` の `mock_http` fixture）、
抽象への依存は `tests/fakes.py` のフェイクを使ってテストする。
リトライのテストでは実際に待たないよう、クライアントには `retry.NO_RETRY` を渡すか、`send_with_retry` に `sleep` を注入する。
`app.py` はカバレッジの計測対象外なので、UI の振る舞いは `tests/test_app.py`（`AppTest`）でテストする。

## アーキテクチャ

```text
app.py (Streamlit UI)
  └─ factory.py (Composition Root: Settings → 具象クライアント)
  └─ flow.py    DecisionFlow: decide → should_stop → generate / stream
       ├─ router.py  DecisionRouter: Confidence Gate（決定論的ポリシー）
       │    └─ DecisionClient ← jev_client.JevClient / mock_jev_client.MockJevClient
       └─ ChatClient (+ StreamingChatClient) ← vllm_client.VLLMClient
evaluation.py  評価 CLI: evals/routing_cases.jsonl → DecisionRouter → 正解率
retry.py       RetryPolicy / send_with_retry（Jev・vLLM クライアントが共通で使う）
```

| モジュール | 責務 |
|---|---|
| `interfaces.py` | レイヤー間の抽象（`DecisionClient` / `ChatClient` / `StreamingChatClient` Protocol と入出力 dataclass、例外） |
| `routes.py` | `Route` と `DEFAULT_ROUTES`（Choice 名・選択基準・System Prompt） |
| `router.py` | `DecisionRouter`。判断結果に Confidence Gate を適用して `RouteDecision` を返す |
| `flow.py` | `DecisionFlow`。判断 → 停止判定 → 生成（通常・ストリーミング）のオーケストレーションとレイテンシ計測 |
| `jev_client.py` | Jev API（`POST /v1/systemone`）の実装 |
| `mock_jev_client.py` | キーワードベースの UI 確認用 Mock |
| `vllm_client.py` | vLLM OpenAI 互換 API の実装（通常・ストリーミング） |
| `retry.py` | HTTP リトライポリシー（指数バックオフ、`Retry-After` 対応） |
| `evaluation.py` | ルーティング精度の評価（`jev-router-eval`） |
| `config.py` | 環境変数 → `Settings` |
| `factory.py` | `Settings` から具象クライアントを生成する唯一の場所 |
| `app.py` | Streamlit の表示と入力のみ。ビジネスロジックを書かない |

## vLLM とのインターフェース

生成レイヤーとの境界は `interfaces.py` の `ChatClient`（必須）と `StreamingChatClient`（任意）だけ。

```python
class ChatClient(Protocol):
    def chat(self, request: ChatRequest) -> ChatResponse: ...


class StreamingChatClient(Protocol):
    def stream(self, request: ChatRequest) -> ChatStream: ...


@dataclass(frozen=True)
class ChatRequest:
    system_prompt: str  # Route.system_prompt
    user_prompt: str  # ユーザー入力
    temperature: float = 0.2
    max_tokens: int = 1200


@dataclass(frozen=True)
class ChatResponse:
    content: str  # 生成されたテキスト
    model: str  # 実際に使われたモデル名


@dataclass(frozen=True)
class ChatStream:
    model: str
    chunks: Iterator[str]  # テキスト断片
```

失敗時は必ず `ChatError` を送出する（httpx の例外を外に漏らさない）。ストリーミングでは、接続時の失敗は `stream()` から、反復中の失敗は `chunks` の反復から送出する。
`StreamingChatClient` を実装しないバックエンドでも、`DecisionFlow.stream` が 1 断片として扱うので UI は動く。

`VLLMClient` が呼び出す vLLM の OpenAI 互換エンドポイント（`base_url` は `/v1` まで含む）：

| メソッド | パス | 用途 |
|---|---|---|
| `GET` | `{base_url}/models` | `model` 未指定時に先頭のモデル ID を取得（インスタンス内でキャッシュ） |
| `POST` | `{base_url}/chat/completions` | `messages=[system, user]`, `temperature`, `max_tokens` を送り、`choices[0].message.content` を返す |
| `POST` | `{base_url}/chat/completions`（`stream: true`） | SSE の `data: {...}` から `choices[0].delta.content` を順に返す。`data: [DONE]` で終了 |

すべての呼び出しは `retry.send_with_retry` を通す（408・429・5xx・接続エラーを指数バックオフでリトライ）。

認証は `Authorization: Bearer {VLLM_API_KEY}`（vLLM のデフォルトは `EMPTY`）。

**ルール:**
- vLLM 以外の LLM（Ollama、OpenAI など）に対応するときは、`ChatClient` を満たす新しいクラスを追加し、`factory.build_chat_client` で切り替える。`VLLMClient` に分岐を追加しない。
- `router.py` / `flow.py` / `app.py` から `VLLMClient` を直接 import しない。必ず `ChatClient` に依存する。
- リクエストやレスポンスの項目を増やすときは、`ChatRequest` / `ChatResponse` を拡張し、`tests/test_vllm_client.py` で実際に送る payload を検証する。

## SOLID 原則

このプロジェクトは SOLID 原則に従って開発する。

- **S（単一責任）**: 1 モジュール・1 クラスにつき責務は 1 つ。UI（`app.py`）、ポリシー（`router.py`）、外部 API 呼び出し（`*_client.py`）、設定（`config.py`）、組み立て（`factory.py`）を混ぜない。
- **O（開放閉鎖）**: ルートの追加は `routes.py` に `Route` を追加するだけで済むようにする。新しいバックエンドは新しいクラスとして追加し、既存クラスに `if` 分岐を足さない（例：Mock は `mock_mode` フラグではなく `MockJevClient` として独立させている）。
- **L（リスコフの置換）**: `DecisionClient` / `ChatClient` の実装は、同じ入出力の契約と例外の契約（`DecisionError` / `ChatError`）を守り、互いに差し替えられること。
- **I（インターフェース分離）**: Protocol は小さく保つ（現在はどれもメソッド 1 つ）。必要のないメソッドを実装に強制しない（例：ストリーミングは `ChatClient` に足さず、`StreamingChatClient` として分けている）。
- **D（依存性逆転）**: 上位モジュールは `interfaces.py` の抽象に依存する。具象クラスを知っているのは `factory.py` とテストだけ。外部 I/O（`httpx.Client` など）はコンストラクタで注入できるようにする。

## コーディング規約

- Python 3.12 以上。型ヒントを必ず付ける。`from __future__ import annotations` を先頭に書く。
- 値オブジェクトは `@dataclass(frozen=True)` を使う。
- import はパッケージからの絶対 import（`from jev_decision_router.xxx import ...`）にする。
- lint / フォーマットは ruff（設定は `pyproject.toml`、行長は 100）。型チェックは mypy の strict モード。
- `.env` はコミットしない。新しい環境変数を追加したら `config.Settings`・`.env.example`・README の表を更新する。
- ルートや System Prompt を変えたら、`uv run jev-router-eval` で精度が落ちていないか確認する。精度を測りたいケースは `evals/routing_cases.jsonl` に追加する。
- 作業の進み具合は `docs/todo.md` で管理する。
