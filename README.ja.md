# Sorta

[![CI](https://github.com/shinKatana0/sorta/actions/workflows/check.yml/badge.svg)](https://github.com/shinKatana0/sorta/actions/workflows/check.yml)
[![Release](https://img.shields.io/github/v/release/shinKatana0/sorta)](https://github.com/shinKatana0/sorta/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> 言語: [English](README.md) · [Русский](README.ru.md) · **日本語**

**大容量の写真・動画コレクション**（60 GB 超で動作確認、300 GB 超を想定した設計）を
**都市/国**・**人物**・**イベント**単位で整理し、きれいなフォルダ構成に**振り分ける**
ツールです。安全性を最優先: デフォルトは dry‑run、移動はジャーナルに記録され、
1コマンドで元に戻せます。

Sorta は**すべてローカルで動作**し（顔・シーン・テキストの ML モデルはお使いの
マシン上で実行）、**元の写真ファイルを一切変更しません**。**CLI** と、手順に沿って
操作できる**ローカル Web アプリ**の両方を用意しています。

![Sorta — ローカル Web アプリ: 都市ごとの整理、フォルダ言語の切り替え、重複レビュー、取り消し](docs/assets/hero.gif)

<sub>合成のデモコレクションでのローカル Web アプリ（`sorta ui`）— 都市ツリー、フォルダ言語の即時切り替え（en/ru/ja）、重複レビュー。</sub>

> ⚡ **本格的に使うなら** — 顔認識、深い VLM ティア、大規模コレクション —
> **NVIDIA GPU（CUDA 13）VRAM ≥ 4 GB** を推奨します（VLM ティアは **≥ 8 GB**）。
> CPU でもすべて動作しますが、これらの処理は明らかに遅くなります —
> [システム要件](#システム要件)を参照してください。

> 📖 **ユーザーガイド:** [English](docs/guide/user-guide.en.md) ·
> [Русский](docs/guide/user-guide.ru.md) · [日本語](docs/guide/user-guide.ja.md)

---

## 主な機能

- **都市/人物/イベント別の振り分け**を単一のインデックスから — モード切り替えに
  再スキャンは不要。
- **オフライン位置情報解決**（同梱の GeoNames）+ GPS とセッション推定による補完。
  オンライン Nominatim/OSM もオプションで利用可能。
- **デフォルトは軽量な基本実行:** `sorta run` / UI の **処理** ボタンは、都市別
  振り分け + 重複検出（index → geo → landmarks → junk → 近似重複ハッシュ）だけを
  行います。**顔検出とイベント検出は opt‑in**（`--faces`/`--events`、または対応
  するチェックボックス）— 最も時間のかかる処理で、誰もが必要とするわけではない
  ためです。
- **顔と人物:** 有効にすると、ローカルでの検出 + クラスタリング（insightface）。
  クラスタに名前を付けて統合し、人物別の振り分けやアルバム作成に利用できます。
- **イベント:** 有効にすると、時間の間隔 + 都市によるクラスタリングと、
  ローカライズされた名前付け。手動イベントも作成可能。
- **重複検出:** 完全一致（blake3）と近似重複（知覚ハッシュ）。まとめて確認できる
  UI 付き。
- **ゴミ・書類・商品の分類:** スクリーンショットやミームは別枠に、書類（パスポート・
  レシート・医療書類など）はレビュー用フォルダ `_書類/` に収集
  （CLIP + テキスト密度による判定）。深い VLM ティアを有効にすると、販売用の商品
  写真は `_商品/` へ。
- **アルバム:** 特定の人物/イベントの写真だけを **ハードリンク**（追加容量ほぼ
  ゼロ）・コピー・移動のいずれかで名前付きフォルダに集約。
- **ローカル Web アプリ**（`sorta ui`）: フォルダの処理、プランの確認、重複の
  解消、人物の命名、振り分け/アルバムの適用まで、すべてブラウザ上で完結。
  **概要** タブはコレクションの状態を 1 画面にまとめ、数値はクリックできます。
  **個人写真ではないもの** タブでは商品・書類・スクリーンショット・ミームを確認し、
  まとめて写真へ戻せます。**人物**/**イベント** タブは、実際にそのステージを
  実行した後にのみ表示されます。
- **3言語対応** の UI とフォルダ名: **ru / en / ja**。
- **安全性を重視した設計:** dry‑run、ジャーナル + `undo`、blake3 での検証、
  既存ファイルを上書きしない（`_1`、`_2` の接尾辞）。

---

![「都市」タブ — 移動前に確認する、提案されたフォルダ構成（国 / 都市 / 年）](docs/assets/process.png)

## システム要件

| | CPU プロファイル（`cpu` エクストラ） | GPU プロファイル（`gpu` エクストラ） |
|---|---|---|
| ハードウェア | 任意の x86‑64 マシン | NVIDIA GPU + **CUDA 13** 対応ドライバ |
| VRAM | 不要 | **~3 GB** 基本 + 顔検出（RTX 5090 で実測: CLIP ViT‑L 2.0 GB + buffalo_l 0.6 GB）— **≥ 4 GB** で快適、深い VLM ティアには **≥ 8 GB**（Qwen2.5‑VL‑3B、約 7 GB 見込み） |
| 顔検出/CLIP の速度 | 動作するが**低速**（顔検出・イベント検出を有効にした大規模コレクションでは数時間） | 高速 — 2026‑07‑28 実測: 24,196 枚の写真、顔検出+イベント+junk で深いティア無しなら ≈ **40分**。オプションの深い VLM ティアは **+122分**（一度きり） |
| 向いている用途 | 任意のマシンでの都市別振り分け + 重複検出、顔検出/イベントを有効にした小規模コレクション | 顔検出/イベントを常用する大規模コレクション（300 GB 超） |

両プロファイル共通: Python **3.11–3.14**、[`uv`](https://docs.astral.sh/uv/)、
そして **PATH 上の `exiftool`**（HEIC/RAW/動画のメタデータに必須 — ないと
Pillow にフォールバックし、JPEG/PNG/TIFF/WEBP のみ読み取り可能で動画は読めません）。
インデックス（SQLite）とサムネイル用のディスク容量はコレクションのサイズに応じて
増加します。`--copy` での振り分けはコレクションサイズの概ね ×2、`--link`
（ハードリンク）はほぼ追加容量なしで済みます。

上記のタイミングは当方の環境での参考値であり、保証するものではありません。
RAM/VRAM に関する詳細を含む完全な内訳は
[ユーザーガイド](docs/guide/user-guide.ja.md#2-必要要件) を参照してください。

---

## クイックスタート

```bash
# 一度だけインストール — 以下から 1 つを選びます。エクストラはパッケージ指定の
# 「内側」に書き、ハードウェアプロファイルを決めます（`cpu` と `gpu` は排他）。
uv tool install "C:\path\to\sorta[cpu]"       # NVIDIA GPU なし — `sorta` を PATH に追加
uv tool install "C:\path\to\sorta[gpu]"       # NVIDIA GPU + CUDA 13 ドライバ
uv tool install "C:\path\to\sorta[gpu,vlm]"   # ...深い VLM ティアも
uv tool install -e "C:\path\to\sorta[gpu]"    # editable — ローカルでコードを触る場合

sorta doctor                            # 実際に何が入ったかを確認（最初にこれ）
cp config.example.yaml config.yaml      # `sources` と `language` を設定
# exiftool は HEIC/RAW/動画に必須 — 先にインストールしてください（「要件」参照）

# 一番簡単: Web アプリ
sorta ui                                # http://127.0.0.1:8756 → フォルダを処理 → 確認

# または CLI
sorta index /path/to/photos             # スキャン
sorta run                               # geo, landmarks, junk + 近似重複ハッシュ（都市+重複）
sorta run --faces --events              # ...顔検出とイベント構築も行う
sorta sort --by city --dest /path/to/sorted            # dry-run プラン（CSV + HTML）
sorta sort --by city --dest /path/to/sorted --copy --apply   # 適用（copy は非破壊）
sorta undo                              # 必要なら直前のバッチを取り消し
```

> **`uv tool install` に `--extra` フラグはありません** — エクストラは上記のとおり
> 引用符付きのパッケージ指定の内側に書きます。付け忘れると、GPU 搭載機でも黙って
> **CPU プロファイル**（`torch==2.13.0+cpu`）が入ります。それを見つける手段が
> `sorta doctor` です: torch のビルド、onnxruntime のプロバイダ、地理データベース、
> ログとプレビューキャッシュのパスを表示します。

コードを開発する場合は、プロジェクトの venv を使ってください（`uv sync --extra
gpu --extra dev` の後にアクティベートし、同じ `sorta …` コマンドをそのまま実行
すればコードの変更が即反映されます）。すべてのインストール方法、`sorta doctor` の
出力の読み方、`onnxruntime`/`onnxruntime-gpu` の落とし穴、そして素の
`uv run sorta …` がその 1 つでは*ない*理由は、ユーザーガイドの
[「インストール」](docs/guide/user-guide.ja.md#3-インストール)を参照してください。

大きな実行の前に知っておくとよいことが 2 つあります。Sorta は**プレビュー
キャッシュ**（フレームごとに 1 回だけデコード、1 枚あたり約 150 KB — 大規模
コレクションではギガバイト単位。確認は `sorta cache`、削除は `sorta cache --clear`）
を持ち、段階ごとの所要時間を含む**実行ログ**を
`%LOCALAPPDATA%\sorta\logs\sorta.log`（他 OS では `~/.cache/sorta/logs/sorta.log`）
に書き出します。詳細はユーザーガイドにあります。

実際のコマンド出力付きの完全なウォークスルー、コマンドリファレンス、設定
リファレンスは [ユーザーガイド](docs/guide/user-guide.ja.md) にあります。

---

## 安全性とプライバシー

- **元ファイルは一切変更されません。** 振り分けはファイルの移動/コピーのみ行い、
  EXIF は書き換えません。
- **デフォルトは dry‑run。** すべての操作は実行前にジャーナルへ記録され、
  `sorta undo` で取り消せます。
- **デフォルトはローカル処理。** ML はすべてお使いのマシン上で実行されます。
  オンラインプロバイダ（Nominatim ジオコーディング: GPS 座標のみ送信、
  Claude API によるイベント命名: 明示的に有効にした場合のみ、イベントごとの
  サンプル写真を数枚送信）はデフォルトで無効です。それぞれが何を送信するかの
  詳細は [SECURITY.md](SECURITY.md) を参照してください。
- **書類**（パスポート、レシート、医療書類など）はローカルのレビュー用フォルダ
  `_未分類/文書/` に収集され、お使いのマシン上でのみ処理されます。
- Web アプリは `127.0.0.1` のみで待ち受けます。

詳細は [SECURITY.md](SECURITY.md) を参照してください。

---

## ドキュメント

- **[ユーザーガイド](docs/guide/user-guide.ja.md)** — インストール、設定、
  ワークフロー、コマンド/設定リファレンス、トラブルシューティング（EN / RU / JA）
- `docs/ARCHITECTURE.md` — アーキテクチャ、モジュールの所有権、データ契約
- `CONTRIBUTING.md` — 貢献方法 · `SECURITY.md` — プライバシーと報告 ·
  `NOTICE` — サードパーティデータの帰属表示（GeoNames、OpenStreetMap/Nominatim）

---

## 開発

```bash
uv sync --extra cpu --extra dev         # または --extra gpu
uv run python scripts/check.py          # ゲート: ruff + mypy + pytest（カバレッジ込み）
```

完全なセットアップとゲートの詳細は [CONTRIBUTING.md](CONTRIBUTING.md) を参照して
ください。

## ライセンス

MIT — [LICENSE](LICENSE) を参照。使用/参照するサードパーティの地理データには
独自の帰属表示要件があります — [NOTICE](NOTICE) を参照してください。
