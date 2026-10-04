# TODO

完了したら `- [ ]` を `- [x]` に変えてください。

## 完了

### プロジェクト基盤

- [x] コードを `src/` に移動し、`src/jev_decision_router/` パッケージ構成にする
- [x] uv ベースに移行する（`pyproject.toml` / `uv.lock`、`requirements.txt` は廃止）
- [x] `.env.example` を作成する
- [x] `.gitignore` を作成する（`.env`・`.venv/`・キャッシュ類を除外）
- [x] 初回コミットを作り、GitHub に push する

### 設計・品質

- [x] SOLID 原則に沿ってリファクタリングする（`interfaces` / `routes` / `router` / `flow` / `factory` などに分割）
- [x] vLLM とのインターフェースを `ChatClient` Protocol として明示する
- [x] Mock を独立したクラス（`MockJevClient`）にする
- [x] pytest を導入する（41 件：クライアント・ルーター・フロー・設定・Streamlit UI）
- [x] ruff（lint / format）を導入する
- [x] `CLAUDE.md` に開発ルール（SOLID・テスト必須・lint・vLLM の境界）を書く

### ドキュメント

- [x] README を整理する（構成・設定一覧・vLLM のインターフェース・開発コマンド）

### 動作確認

- [x] Mock モードで Streamlit を起動する
- [x] vLLM（`Qwen/Qwen2.5-1.5B-Instruct`）を起動し、判断 → 生成が最後まで動くことを確認する

## 未完了

### 動作確認

- [ ] 実際の Jev API で動作を確認する（`TYPESAFE_API_KEY` が必要）
- [ ] Jev API の仕様（`POST /v1/systemone` のリクエストとレスポンスの形式）が `jev_client.py` の実装と一致しているか、公式ドキュメントで確認する
- [ ] 4 つのルート（`direct_answer` / `deep_analysis` / `coding` / `human_review`）すべてについて、ブラウザで一通り試す

### ドキュメント

- [ ] README に、8GB 程度の GPU 向けの vLLM 起動例を追記する（1.5B モデル、`VLLM_USE_FLASHINFER_SAMPLER=0`、`--max-model-len`、`--gpu-memory-utilization`）
- [ ] デモ画面のスクリーンショットを README に載せる

### 品質・CI

- [ ] GitHub Actions で `ruff check`・`ruff format --check`・`pytest` を自動実行する
- [ ] 型チェック（mypy または pyright）を導入する
- [ ] カバレッジを計測し、目標値を決める（`pytest --cov`）
- [ ] pre-commit で ruff を自動実行する

### 機能改善

- [ ] vLLM の回答をストリーミング表示にする
- [ ] Jev と vLLM のエラー時に、リトライやタイムアウトの設定をできるようにする
- [ ] ルーティング精度を評価するためのサンプル入力と期待ルートのデータセットを作る
- [ ] Streamlit 起動時の static フォルダの警告を解消する（`server.enableStaticServing=false` を設定する）
