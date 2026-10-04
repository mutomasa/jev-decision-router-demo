# jev-decision-router-demo

**Jev を意思決定レイヤー（System 1）、ローカル vLLM を推論・生成レイヤー（System 2）として組み合わせた Streamlit デモです。**

ユーザーの自然言語リクエストを Jev の `Choice` で分類し、Confidence Gate を通過した結果に応じて、ルートごとの System Prompt でローカル vLLM を呼び出します。
Agentic AI / Agentic Mesh における **Decision Plane（意思決定レイヤー）** の最小 PoC として利用できます。

## 目次

- [Concept](#concept)
- [Architecture](#architecture)
- [Decision Choices](#decision-choices)
- [Confidence Gate](#confidence-gate)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Run](#run)
- [Development](#development)

---

## Concept

一般的な LLM アプリでは、意図理解・タスク分類・Agent/Tool 選択・リスク判定・推論・回答生成をすべて 1 つの LLM に任せがちです。
本プロジェクトでは「判断」と「生成」を分離し、その間に決定論的なソフトウェアポリシーを挟みます。

| レイヤー | 担当 | 役割 |
|---|---|---|
| System 1 | **Jev** | 高速な判断（どの処理を選ぶか） |
| Policy | **Python** | 決定論的な制御（Confidence Gate、ルート選択） |
| System 2 | **vLLM**（Qwen / Llama など） | 熟考・推論・文章 / コード生成 |

```text
Probabilistic AI Decision  →  Deterministic Software Policy  →  Execution
         (Jev)                        (router.py)                  (vLLM)
```

---

## Architecture

```text
              User Request
                    │
                    ▼
          ┌──────────────────┐
          │       Jev        │  Choice / Probability / Confidence
          │  Decision Layer  │
          └────────┬─────────┘
                   ▼
          ┌──────────────────┐
          │ Software Policy  │  Confidence Gate / Route Selection
          └────────┬─────────┘
          ┌────────┴─────────┐
          ▼                  ▼
     Local vLLM         Human Review
  (route別 System Prompt)   (自動生成を停止)
          │
          ▼
       Response
```

### 例

```text
FastAPIでJWT認証付きAPIを実装したい。最小構成のコードを書いてください。
```

Jev の判断結果（イメージ）：

```json
{
  "choice": "coding",
  "confidence": 0.94,
  "probabilities": {
    "direct_answer": 0.01,
    "deep_analysis": 0.04,
    "coding": 0.94,
    "human_review": 0.01
  }
}
```

→ `coding` 用 System Prompt でローカル vLLM を呼び出し、コードを生成します。

---

## Decision Choices

ルート定義は `src/jev_decision_router/routes.py` の `DEFAULT_ROUTES` にあります。

| Choice | 役割 | 入力例 |
|---|---|---|
| `direct_answer` | 単純な質問や短い説明 | 「HTTP 404 とは？」 |
| `deep_analysis` | 比較・設計・調査・複雑な分析 | 「売上悪化の原因を整理し、打ち手を比較して」 |
| `coding` | コード生成・実装・デバッグ | 「Python で CSV を読み込むコードを書いて」 |
| `human_review` | 高リスク・不可逆・情報不足で人間判断が必要 | 「顧客の本番 DB を削除して」 |

---

## Confidence Gate

Jev の判断をそのまま実行せず、Python のポリシーで検査します。

- `confidence < threshold`（デフォルト `0.65`）の場合、ルートを `human_review` に切り替える
- Jev が `DEFAULT_ROUTES` に存在しない Choice を返した場合も `human_review` に倒す
- `human_review` になった場合、サイドバーの **Stop before vLLM on human_review** が ON（デフォルト）なら vLLM を呼ばずに停止する。OFF の場合はレビュー用 System Prompt で vLLM を呼び出し、確認ポイントを整理させる

```text
choice = coding, confidence = 0.58, threshold = 0.65
  → confidence 不足 → human_review
```

---

## Project Structure

```text
.
├── .env.example
├── CLAUDE.md                    # 開発ルール（SOLID / テスト / lint）
├── pyproject.toml               # uv / pytest / ruff の設定
├── uv.lock
├── src/jev_decision_router/
│   ├── app.py                   # Streamlit UI（表示と入力のみ）
│   ├── interfaces.py            # DecisionClient / ChatClient などの抽象インターフェース
│   ├── routes.py                # Route 定義（Choice・選択基準・System Prompt）
│   ├── router.py                # DecisionRouter（Confidence Gate）
│   ├── flow.py                  # DecisionFlow（判断 → 停止判定 → 生成）
│   ├── jev_client.py            # Jev API クライアント（POST /v1/systemone）
│   ├── mock_jev_client.py       # UI 確認用の Mock クライアント
│   ├── vllm_client.py           # vLLM OpenAI 互換 API クライアント
│   ├── config.py                # 環境変数 → Settings
│   └── factory.py               # Settings から各クライアントを生成
└── tests/                       # pytest
```

### vLLM とのインターフェース

生成レイヤーとのやり取りは `interfaces.ChatClient` だけを通して行います。

```python
class ChatClient(Protocol):
    def chat(self, request: ChatRequest) -> ChatResponse: ...
```

| 型 | フィールド |
|---|---|
| `ChatRequest` | `system_prompt`, `user_prompt`, `temperature=0.2`, `max_tokens=1200` |
| `ChatResponse` | `content`, `model` |

`VLLMClient` はこれを vLLM の OpenAI 互換 API に変換します。

| メソッド | パス | 用途 |
|---|---|---|
| `GET` | `{VLLM_BASE_URL}/models` | `VLLM_MODEL` が空のとき、先頭のモデルを自動選択する |
| `POST` | `{VLLM_BASE_URL}/chat/completions` | `messages=[system, user]` を送り、`choices[0].message.content` を返す |

失敗した場合は `ChatError` を送出します。別の LLM バックエンドを使う場合は、`ChatClient` を満たすクラスを追加し、`factory.build_chat_client` で切り替えます。

---

## Setup

### 1. 依存関係のインストール

```bash
git clone <repository-url>
cd jev-decision-router-demo

uv sync
```

[uv](https://docs.astral.sh/uv/) が必要です。`uv sync` を実行すると `.venv` が作成され、開発用の依存（pytest / ruff）も含めてインストールされます。

### 2. ローカル vLLM の起動

vLLM の OpenAI 互換サーバを起動します（例：Qwen）。

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name local-qwen
```

API は `http://localhost:8000/v1` で提供されます。Llama / Mistral / DeepSeek など vLLM 対応モデルであれば利用できます。

---

## Configuration

`.env.example` をコピーして `.env` を作成し、`TYPESAFE_API_KEY` などを編集します。設定値はアプリのサイドバーからも変更できます。

```bash
cp .env.example .env
```

| 変数 | デフォルト | 説明 |
|---|---|---|
| `TYPESAFE_API_KEY` | （なし） | Jev の API キー。Mock Mode OFF 時は必須 |
| `JEV_BASE_URL` | `https://api.typesafe.ai` | Jev API の URL |
| `JEV_MODEL` | `jev-latest` | Jev モデル名 |
| `JEV_MOCK_MODE` | `false` | `true` でキーワードベースの Mock 判定を使用 |
| `JEV_CONFIDENCE_THRESHOLD` | `0.65` | Confidence Gate のしきい値 |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM の OpenAI 互換 API の URL |
| `VLLM_MODEL` | （空） | 空の場合は `/v1/models` の先頭モデルを自動選択 |
| `VLLM_API_KEY` | `EMPTY` | vLLM の API キー |

### Mock Mode

Jev の API キーがなくても UI やルーティングの動作を確認できます。
`JEV_MOCK_MODE=true` にすると、キーワードマッチによる決定論的な判定を返します。**UI 確認用であり、Jev の代替ではありません。**

---

## Run

vLLM を起動した状態で、Streamlit を起動します。

```bash
uv run streamlit run src/jev_decision_router/app.py
```

ブラウザで <http://localhost:8501> にアクセスします。

### 処理フロー

1. **User Request** — テキストを入力し「Run decision flow」を押す
2. **Jev Decision** — Choice / Probability / Confidence を取得
3. **Policy Evaluation** — Confidence Gate でルートを確定（不足時は `human_review`）
4. **Execution** — ルート別 System Prompt でローカル vLLM を呼び出す（`human_review` は停止）
5. **Response** — 回答と Jev / vLLM それぞれのレイテンシを表示

---

## Development

開発ルールの詳細は [CLAUDE.md](CLAUDE.md) を参照してください。

```bash
uv run pytest                                          # テスト
uv run ruff check . && uv run ruff format --check .    # lint / フォーマット確認
uv run ruff check --fix . && uv run ruff format .      # 自動修正
```
