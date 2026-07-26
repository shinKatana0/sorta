# Sorta — ユーザーガイド（日本語）

> 言語: [English](user-guide.en.md) · [Русский](user-guide.ru.md) · **日本語**

Sorta は、大規模な写真・動画コレクション（60 GB 以上で検証、300 GB 以上を想定）を
**インデックス化**し、ファイルを新しいフォルダ構成へ**振り分ける**コマンドライン／
ローカル Web ツールです。**都市・国別**、**人物別**、**イベント別**に整理でき、安全性
を最優先します（既定はドライラン、移動ジャーナル、ワンコマンドでの取り消し）。

- **既定でローカル動作。** すべての ML モデル（顔、シーン／テキスト検出）は自分の
  マシン上でオフライン実行されます（GPU 推奨）。設定でオンラインプロバイダーを明示的
  に有効化しない限り、外部へ送信しません。
- **オリジナルは決して変更しません。** 振り分けはファイルを*移動*または*コピー*する
  だけで、EXIF は書き換えません。`--copy`／`--link` なら元ファイルはそのまま残ります。
- **2 つの使い方:** ガイド付き **Web UI**（`sorta ui`）と **CLI**。どちらも同じエンジン
  のラッパーなので、好きな方を選べます。

---

## 1. 目次

1. [必要要件](#2-必要要件)
2. [インストール](#3-インストール)
3. [設定](#4-設定)
4. [基本概念](#5-基本概念)
5. [クイックスタート — Web UI（推奨）](#6-クイックスタート--web-ui推奨)
6. [クイックスタート — CLI](#7-クイックスタート--cli)
7. [処理パイプライン](#8-処理パイプライン)
8. [振り分け：都市・人物・イベント](#9-振り分け都市人物イベント)
9. [重複](#10-重複)
10. [人物と顔クラスタ](#11-人物と顔クラスタ)
11. [イベント](#12-イベント)
12. [アルバム](#13-アルバム)
13. [不要写真・スクショ・書類](#14-不要写真スクショ書類)
14. [安全性・取り消し・プライバシー](#15-安全性取り消しプライバシー)
15. [コマンド一覧](#16-コマンド一覧)
16. [メンテナンスと診断](#17-メンテナンスと診断)
17. [プレビューキャッシュ](#18-プレビューキャッシュ)
18. [実行ログ](#19-実行ログ)
19. [モデルのオフライン動作](#20-モデルのオフライン動作)
20. [設定リファレンス](#21-設定リファレンス)
21. [トラブルシューティング](#22-トラブルシューティング)

---

## 2. 必要要件

| 項目 | 要件 |
|---|---|
| OS | Windows、Linux、macOS |
| Python | 3.11 – 3.14（`requires-python >=3.11,<3.15`） |
| 環境マネージャ | [`uv`](https://docs.astral.sh/uv/)（推奨）または `pip` |
| `exiftool` | **必須**（HEIC/RAW/動画のメタデータ: 日付・GPS・向き — 事実上どのスマホ写真にも該当）。無い場合は Pillow にフォールバックしますが、読めるのは JPEG/PNG/TIFF/WEBP のみで動画は読めません。 |
| ディスク空き | 新構成に十分な容量。`--copy` はデータを複製（×N）。`--link`（ハードリンク）はほぼ追加容量ゼロ（同一ボリューム、NTFS/ext4/APFS）。加えて SQLite インデックスと（任意で）サムネイル ― どちらも写真コレクション本体に比べれば小さい。 |

Sorta の ML バックエンド（顔検出、不要写真分類用の CLIP/OCR）は、**互いに排他的な
2 つのインストールプロファイル**のいずれかで導入します — ハードウェアに合わせて
選んでください:

| | CPU プロファイル（`cpu` エクストラ） | GPU プロファイル（`gpu` エクストラ） |
|---|---|---|
| ハードウェア | 任意の x86‑64 マシン、GPU 不要 | NVIDIA GPU + **CUDA 13** 対応ドライバ（Blackwell/RTX 5090 で検証済み） |
| バックエンド | `onnxruntime`（CPU）+ CPU ビルドの torch/torchvision | `onnxruntime-gpu` + CUDA 13/cuDNN 9 ランタイム（pip wheel）+ CUDA ビルドの torch/torchvision |
| 顔検出/CLIP の速度 | 正しく動作するが**低速** — 大規模コレクションでは `faces`/`junk`/`landmarks` に数時間かかることも。都市別振り分け+重複検出には十分（そもそも顔検出/イベントは opt‑in、§8 参照）。顔検出/イベントを有効にした小規模コレクションでも使える。 | 高速。当方の 6,298 枚のテストコレクションでの参考値（顔検出+イベント+junk 有効）: **≈ 45分**（高速/CLIP ティア）、オプションの深い VLM ティアで **≈ 77分**（`naming.vlm_enabled` / `uv sync --extra vlm`）。 |
| RAM | 8 GB 以上を推奨（インデックス作成/ハッシュ計算が最も RAM を使う部分で、プロファイルによらず共通） | 同上、加えて GPU ドライバが確保する分 |
| VRAM | 該当なし | ベース + 顔認識で **~3 GB**（RTX 5090 で実測: CLIP ViT‑L ≈2.0 GB + buffalo_l ≈0.6 GB）— **4 GB 以上**の GPU が快適。オプションの深い VLM ティア（Qwen2.5‑VL‑3B）は ≈7 GB 追加（3B fp16 モデルからの推定、未実測）→ 合計 **8 GB 以上** |

上記のタイミングと VRAM の数値は当方の環境での観測値であり、保証ではありません —
実際の値はコレクションの構成（動画プレビュー、RAW ファイル、写真あたりの顔の数が
主なコスト要因）によって変わります。

---

## 3. インストール

### 3.1 インストール前に

```bash
git clone https://github.com/shinKatana0/sorta.git
cd sorta

# exiftool を導入 — HEIC/RAW/動画メタデータに**必須**:
#   Windows: winget install OliverBetz.ExifTool
#   Debian/Ubuntu: sudo apt install libimage-exiftool-perl
#   macOS: brew install exiftool

# テンプレートから設定ファイルを作成
cp config.example.yaml config.yaml
```

`exiftool` が無いと Sorta は Pillow にフォールバックします: 読めるのは
JPEG/PNG/TIFF/WEBP だけで、動画は一切読めず、HEIC/RAW からは日付・GPS・向きも
取得できません。スマホのコレクションでは、Sorta が振り分けに使うメタデータの
ほとんどがそれに当たります。任意ではなく必須と考えてください。

### 3.2 ハードウェアプロファイルを選ぶ

Sorta の ML バックエンド（顔検出、CLIP/OCR）には、ハードウェア用プロファイルが
ちょうど 1 つ必要です。`cpu` と `gpu` は**互いに排他的な**エクストラです
（`pyproject.toml` の `tool.uv.conflicts`）— どちらか一方だけを入れ、両方は
入れません。各組み合わせで実際に入るもの（2026‑07‑26 に `uv pip compile` の
解決で確認）:

| エクストラ | 入るもの |
|---|---|
| `cpu` | `torch==2.13.0+cpu`、`onnxruntime` |
| `gpu` | `torch==2.13.0+cu130`、`onnxruntime-gpu` |
| `gpu,vlm` | `gpu` の内容 + `transformers==4.51.3` |
| `cpu,vlm` | `cpu` の内容 + `transformers==4.51.3` |

`vlm` は任意の深い VLM 分類ティア（`naming.vlm_enabled` / `--deep`、§8 参照）です。
無い場合、そのティアは黙って高速 CLIP ティアへフォールバックします。`dev` は
品質ゲート用ツール（ruff、mypy、pytest）を追加します — `scripts/check.py` や
テストスイートを実行する場合にのみ必要です。

### 3.3 `sorta` コマンドのインストール（`uv tool install`）

> **`uv tool install` に `--extra` フラグはありません。** エクストラは*パッケージ
> 指定の内側*、つまりパスと同じ引用符付き引数の中に書きます:
> `uv tool install "C:\path\to\sorta[gpu]"`。実際の利用者がつまずいたのはまさに
> ここでした: エクストラ無しで実行した結果、素のパッケージが解決されて
> **CPU プロファイル**が黙って組み上がり、まともな GPU があるマシンなのに torch
> は `+cpu` で入ってしまいました。

```bash
# CPU のみ
uv tool install "C:\path\to\sorta[cpu]"

# GPU（NVIDIA / CUDA 13）
uv tool install "C:\path\to\sorta[gpu]"

# GPU + 深い VLM ティア
uv tool install "C:\path\to\sorta[gpu,vlm]"

# CPU + 深い VLM ティア
uv tool install "C:\path\to\sorta[cpu,vlm]"

# …いずれも editable で（ローカルでコードを触る場合）
uv tool install -e "C:\path\to\sorta[gpu]"
```

`C:\path\to\sorta` はチェックアウトのパスです。すでにその中にいるなら `.` でも
構いません（`uv tool install ".[gpu]"`）。引数全体は必ず引用符で囲んでください —
多くのシェルでは引用されない `[...]` は glob として解釈されます。

- **コードを編集するなら `-e` を使ってください。** editable でないインストールは
  インストール時点のスナップショットです: ファイルを直して `sorta` を実行しても
  古いコピーの挙動のままで、理由を示すものは画面のどこにも出ません。`-e` なら
  編集は即座に反映され、忘れがちな再インストール手順そのものが無くなります。
- **プロファイルの切り替え、または `git pull` 後の非 editable インストールの更新**
  — 目的のエクストラを付けて `--force` で再インストール:
  `uv tool install --force "C:\path\to\sorta[gpu]"`。
- Sorta が PyPI に公開されたら、同じ形が `uv tool install "sorta[gpu]"` になります
  — ローカルのチェックアウトは不要です。

これは `pyproject.toml` のプロファイル/インデックス設定（`pytorch-cu130` /
`pytorch-cpu` インデックス）を `uv sync` と同じように解決し、`sorta` コマンドを
PATH に追加します。以降は、どのターミナルのどのディレクトリからでも `sorta ui`、
`sorta index …` などをそのまま実行できます — `uv run` も、venv のアクティベートも
不要です。

### 3.4 開発用の環境（`uv sync`）

```bash
uv sync --extra gpu --extra dev      # NVIDIA GPU + CUDA 13 ドライバ
# または
uv sync --extra cpu --extra dev      # NVIDIA GPU なし

# シェルセッションごとに一度アクティベート:
.\.venv\Scripts\Activate.ps1         # Windows PowerShell
source .venv/bin/activate            # Linux/macOS/bash
```

`--extra` をフラグとして受け取るのはこの `uv sync` の方で、しかもプロファイルは
同じくらい明示が必要です: `--extra cpu` か `--extra gpu` を必ず自分で指定して
ください（自動では選ばれません）。GPU wheel は CUDA 13 ランタイムを通常の pip
パッケージとして取得するので、システムに CUDA Toolkit を入れる必要はありません。
venv がアクティブな状態では `sorta …` はチェックアウトから直接実行され、コードの
変更は即座に反映されます。

> **`uv run sorta …` を日常的なコマンドとして使わないでください。** `uv run
> <cmd>` は実行のたびに、環境を `pyproject.toml` の基本依存関係セットへ
> 再同期します — そのコマンドで毎回 `--extra <プロファイル>` を繰り返し指定
> しない限り、この再同期は実行のたびに GPU パッケージを黙って外し(torch は
> CPU ビルドにフォールバック)ます。`uv tool install`（§3.3）とアクティベート
> した venv はどちらもこれを完全に回避します — `uv run` 経由の実行ではなく
> 一度だけインストールする理由はまさにここにあります。

### 3.5 インストール直後に — `sorta doctor`

`sorta doctor` は、ほかの何よりも先に実行すべきコマンドです。データベースには
触れず、何もダウンロードせず、インストールが実際にどうなったかだけを報告します:

```
$ sorta doctor
torch: 2.13.0+cu130 (CUDA available: yes, device: NVIDIA GeForce RTX 5090 Laptop GPU)
onnxruntime providers: TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider (CUDA: yes)
mismatch: no
geo data: C:\...\sorta\data\geo\places.tsv (9.2 MB)
Лог прогона: C:\Users\you\AppData\Local\sorta\logs\sorta.log
Кэш превью: C:\Users\you\AppData\Local\sorta\previews
```

各行の読み方:

- **`torch: …（CUDA available: yes|no）`** — 実際に入ったビルド。`gpu` プロファイル
  なら `+cu130` かつ `CUDA available: yes` でなければなりません。`+cpu` ビルドなら、
  エクストラがインストールコマンドに届いていません（§3.3）。
- **`onnxruntime providers: …`** — 顔検出が動ける実行環境。`CUDAExecutionProvider`
  が一覧にあれば GPU を使えます。無ければ顔検出は CPU で動きます（§3.6）。
- **`mismatch: yes`** — torch は CPU 専用ビルドなのに onnxruntime には CUDA がある
  状態。顔検出は GPU、CLIP と OCR は黙って CPU という、遅く静かな失敗であり、この
  行はまさにそれを捕まえるために存在します。
- **`geo data: …places.tsv (N MB)`** — 同梱の GeoNames データベース。都市別振り分け
  が座標を解決する相手です。存在し、かつ空でない必要があります。行頭に `⚠` が付き
  FILE NOT FOUND / FILE IS EMPTY と出る場合、すべての座標が空の場所に解決されます。
  対処は再インストール（チェックアウトなら `python scripts/build_geodata.py`）です。
- **`Лог прогона:`**（実行ログ）— 実行ログの出力先（§19）。
- **`Кэш превью:`**（プレビューキャッシュ）— デコード済みプレビューの保存先（§18）。
  末尾の ` (ОТКЛЮЧЁН)` はキャッシュが無効という意味です。

他の CLI メッセージと同じく、最後の 2 つのラベルは固定のロシア語です（§4）。
`sorta doctor` は `config.yaml` を読まないため、表示されるプレビューのパスは既定値
（および環境変数 `SORTA_PREVIEW_DIR`）に基づきます。

### 3.6 既知の落とし穴: `onnxruntime` が `onnxruntime-gpu` を上書きする

`insightface`（顔検出）は素の `onnxruntime` パッケージに依存し、`gpu` エクストラは
`onnxruntime-gpu` を入れます。この 2 つは別々の配布物でありながら*同じ*
`onnxruntime` ディレクトリに展開されるため、後から入った方が勝ちます。勝ったのが
CPU 版だと、顔検出はどこにもエラーを出さないまま静かに CPU へ移ります。

兆候は `sorta doctor` に出ます — プロバイダ一覧に `CUDAExecutionProvider` が
**無い**状態です。例:

```
onnxruntime providers: AzureExecutionProvider, CPUExecutionProvider (CUDA: no)
```

回避策は、pip に依存関係を再解決させずに GPU パッケージを上書き再インストールする
ことです:

```bash
python -m pip install --force-reinstall --no-deps onnxruntime-gpu
```

`--no-deps` は省略できません。付けないと pip が `insightface` を再解決し、CPU 版
`onnxruntime` をまた引き戻します。Sorta がインストールされている環境（アクティブ
にしたプロジェクト venv）で実行し、その後 `sorta doctor` をもう一度実行して
`CUDAExecutionProvider` が一覧に戻ったことを確認してください。

---

## 4. 設定

Sorta は `config.yaml` を読み込みます（`config.example.yaml` からコピー）。必ず確認
すべき 2 項目:

```yaml
sources:
  - "D:/Photos"          # 写真・動画のフォルダ（再帰的にスキャン）
database: "sorta.db"     # SQLite インデックスの保存先
language: ja             # UI／フォルダ言語: ru | en | ja（既定 ru）
```

- **`sources`** — スキャンするルートフォルダ（複数可）。コマンドラインでも指定でき
  （`sorta index /path/to/photos`）、そちらが設定を上書きします。
- **`language`** — 生成される**フォルダ名**（例: `日本/…` など）と **Web UI** の言語を
  制御。対応: `ru`、`en`、`ja`。

> **注意:** `language` は CLI 自体のコンソールメッセージ（`Готово: +13 новых, ...`
> のような進捗表示）には影響**しません** — これらは設定に関係なく固定のテキスト
> です。フォルダ名と Web UI は完全にローカライズされます。実際の CLI 出力がどう
> 見えるかは §9 の実例を、`????` のように文字化けする場合は §22 を参照してください。

全オプションは[設定リファレンス](#21-設定リファレンス)を参照。

---

## 5. 基本概念

**インデックスと振り分けは別。** まずパイプラインが SQLite **インデックス**（メタ
データ、位置情報、顔埋め込み、クラスタ、イベント、不要写真分類）を埋めます。振り分け
は、そのインデックスの*ビューをファイルシステムに適用*するだけ。モード切替（都市 ↔
人物 ↔ イベント）で再スキャンは不要です。

**既定でドライラン。** `sort` と `album` はプランを表示するだけで、`--apply` を付ける
まで何も書き込みません。必ず先にプランを確認してください。

**ジャーナルと取り消し。** すべての移動／コピー／リンクは操作の*前*にジャーナルへ
記録され、`sorta undo` で直前のバッチを取り消せます。移動前に blake3 ハッシュを照合し、
名前衝突は `_1`、`_2` を付与 — 既存ファイルを上書きしません。

**3 つの転送モード。**
- **move**（`sort` の既定）— ファイルを移動。ディスク上の構成は 1 つ。
- **copy** — 複製。オリジナルは無傷。複数構成が可能だが容量 ×N。
- **link**（ハードリンク、`album` の既定）— 同一バイト列への別名。追加容量ほぼゼロ。
  ボリューム／FS をまたぐ場合は copy にフォールバック。

**正規構成 + アルバム。** 推奨モデルは、都市別の**正規**構成を 1 つ持ち、加えて必要に
応じて**アルバム**（特定の人物／イベント）をハードリンクで別の名前付きフォルダに集める
方式です。

---

## 6. クイックスタート — Web UI（推奨）

Web UI が最も簡単で、ターミナル操作は起動のみです。

```bash
sorta ui                       # http://127.0.0.1:8756 でローカルサーバー起動
```

ブラウザで:

1. **処理** タブ → 写真フォルダのパスを入力。2 つのチェックボックスは既定で**両方
   オフ**: **「顔を検出」**と**「イベントを検出」**— パイプラインの中で最も時間の
   かかる段階で、意図的に opt‑in にしています（§8 参照）。都市別振り分けだけを
   高速に行いたければ両方オフのまま、必要なら該当するものにチェックを入れます。
   **処理する** をクリックすると、パイプラインがバックグラウンドで段階別プログレス
   付きで実行されます（インデックス → 位置情報 → ランドマーク → [顔] → [イベント]
   → 不要写真 → 類似写真 — 顔/イベントはチェックした場合のみ）。タブを閉じても
   処理は継続します。
2. **都市** タブ → 提案された構成を確認（`国/都市/年/地区`）。常に表示されます。
3. **重複** タブ → 類似写真グループを確認。おすすめの残す 1 枚（★）が事前選択済み。
   異なる場合だけ変更し、一度だけ **すべての選択を保存** をクリック。常に表示され
   ます。
4. **人物** タブ → 顔クラスタが存在する場合のみ表示されます（一度でも「顔を検出」
   にチェックを入れたか、`sorta faces` を実行した場合）。クラスタに名前を付け、
   同一人物の重複クラスタを統合。
5. **イベント** タブ → イベントが存在する場合のみ表示されます（一度でも「イベント
   を検出」にチェックを入れたか、`sorta events` を実行した場合）。イベント名を編集。
   任意の人物／イベントを **フォルダに集める** でフォルダ化。
6. **移動** タブ → 振り分け／アルバム適用後、何がどこへ行ったかを確認。常に表示
   されます。

**処理** タブには、顔/イベント検出のほかにもう2つのチェックボックスがあり、
どちらも `config.yaml` を反映し、この実行に限った完全な上書きとして働きます
(チェック=強制オン、未チェック=強制オフ)— CLI の `--deep`/`--no-deep` と
`--geo online`/`--geo offline`(§8)に相当する UI 版です:

- **「詳細分析（VLM）」** — junk/書類の分類に、高速 CLIP ティアの代わりに
  深い VLM ティアを使用します。実際に有効になるのは、要求され(このチェック
  ボックス、`--deep`、または config の `naming.vlm_enabled: true`)かつ
  インストールされている(extra `vlm`、例: `uv tool install ".[gpu,vlm]"` や
  `uv sync --extra gpu --extra vlm --extra dev`)場合の**両方**を満たしたとき
  だけです — この extra が無ければ自動的に高速 CLIP ティアへフォールバック
  し、チェックボックス下のヒントにもそう表示されます。
- **「オンライン位置情報（海外でより正確）」** — この実行では、同梱の
  オフライン GeoNames データの代わりにオンライン Nominatim による逆ジオコー
  ディングを使用します。送信するのは GPS 座標のみで、写真は送信しません
  (§15 参照)。

新規の処理実行で「人物」「イベント」が表示されないのは想定どおりです — その実行で
顔検出/イベント検出を有効にしなかっただけで、何かが壊れたわけではありません。
チェックを入れて再実行するか（あるいは `sorta faces`/`sorta events` を実行すれば）
タブが表示されます。

サーバーは `127.0.0.1` のみ待受（ネットワークからは不可視）。停止は `Ctrl+C`。

---

## 7. クイックスタート — CLI

以下の例は、`sorta` がすでに PATH 上にある(`uv tool install` かアクティベート
した venv 経由、§3 参照)ことを前提にしています。先頭に `uv run` を付けないで
ください — 理由は §3 の警告を参照。

```bash
# 1) フォルダをインデックス化（メタデータ、ハッシュ、完全重複）
sorta index /path/to/photos

# 2) 基本パイプライン（位置情報、ランドマーク、不要写真）+ 類似写真ハッシュ ― 顔/イベントなし
sorta run
sorta phash

# 2b) …顔検出・イベント検出も含める場合（最も遅い段階、§8 参照）:
sorta run --faces --events

# 3) 都市別振り分けをプレビュー（ドライラン — CSV+HTML プランのみ、移動なし）
sorta sort --by city --dest /path/to/sorted

# 4) 適用（copy は非破壊。--copy を外すと MOVE）
sorta sort --by city --dest /path/to/sorted --copy --apply

# 必要なら直前バッチを取り消し
sorta undo
```

### 実例: 最初から最後まで

以下はすべて、小さな合成テストコレクション（EXIF/GPS 付きで生成した JPEG 13枚:
2 日間の「パリ」旅行、イベントになるには少なすぎる「東京」の 1 日、完全重複、
類似写真、スクリーンショット、パイプラインの動作確認だけに使う「顔」のプレース
ホルダー画像 2 枚 — 実在する誰かの写真ではありません）に対して実行した**実際の
コマンド出力**です。何が起きるかを正確に把握できるよう掲載しており、各モードの
完全な解説は §9–§13 で続けます。

```
$ sorta index -c config.yaml
Готово: +13 новых, ~0 обновлено, 0 пропущено, 0 ошибок, 1 дубликатов помечено

$ sorta geo -c config.yaml
Готово: 12 файлов — exact_gps 10, session_inferred 1, unknown 1

$ sorta faces -c config.yaml
Детекция: 12 файлов, 0 лиц, 12 без лиц, 0 ошибок
Кластеры: 0 (лиц в кластерах: 0, шум: 0, имён сохранено: 0)

$ sorta events -c config.yaml
События: 1 авто (7 файлов, имён сохранено: 0), 0 ручных (0 файлов)

$ sorta junk -c config.yaml
Классификация: 12/12 обработано (photo: 11, screenshot: 1)

$ sorta phash -c config.yaml
pHash посчитан для 13 фото. Отчёт: sorta dupes --near

$ sorta stats -c config.yaml
Файлов в индексе: 13 (+0 с ошибками)
  с GPS:            11 (84%)
  дата из exif     : 13 (100%)
  дата из filename : 0 (0%)
  дата из mtime    : 0 (0%)
  дубликатов:       1
Гео (places): 12
  exact_gps       : 10 (83%)
  unknown         : 1 (8%)
  session_inferred: 1 (8%)
```

ここで注目してほしい点（実際の出力どおりで、演出のために書き換えてはいません）:

- **CLI のメッセージは `language` に関係なくロシア語で出力されます** — §4 の注記
  参照。数字が重要な部分です。大まかな訳: *「Готово: +13 новых」*=「完了: 新規
  +13」、*「дубликатов」*= 重複、*「с GPS」*= GPS あり、*「Детекция」*= 検出、
  *「Кластеры」*= クラスタ、*「События」*= イベント、*「Классификация」*= 分類。
- `index` は **13** ファイルを見つけ、後の `stats` も 13 と報告しています —
  完全重複ファイルは（`dup_of` が設定された状態で）**インデックスされます**が、
  独自の場所/イベント/junk 行を持たないため、`geo`/`junk` は **12** と報告します。
- `faces` は 2 枚のプレースホルダー画像から本当に**顔を 0 件**しか見つけていません
  — 実在の写真用顔検出器はフラットなベクター画像には反応しない、というだけの話
  です。だからこそ「顔を 2 件検出し、Alice と名付けました」のような例をここで
  でっち上げませんでした。人物ワークフローが実際の写真でどう見えるかは §11
  参照。
- `events` は **7** ファイル（パリ旅行）から **1** 件のイベントを構築しました。
  東京の 4 ファイルは `events.min_event_size`（5）を下回ったため、イベント別
  振り分け（§9）では `no_event` のフォールバック先に入ります — バグではなく、
  このしきい値の実際の動作を示す良い例です。
- `landmarks` は上に出てきません。このデータでは実行すべきことが何もなかった
  ためです（GPS の無いファイルで、同梱カタログの実在ランドマークに十分近い
  ものが無かった）— 実際に機能する場合の説明は §9 を参照。

---

## 8. 処理パイプライン

`sorta run`（または UI の **処理** ボタン）が以下の段階を順に実行します。各段階は単独
コマンドでもあり、すべて**インクリメンタル**（再実行は新規／変更ファイルのみ処理）:

| 段階 | コマンド | 既定で実行? | 内容 |
|---|---|---|---|
| インデックス | `sorta index [dir]` | 常に | スキャン、EXIF/日付読取、blake3 ハッシュ、完全重複の印付け。 |
| 位置情報 | `sorta geo` | 常に | GPS から場所を解決。GPS 無しは時間的に近い隣接写真から推定（オフライン GeoNames、有効なら オンライン Nominatim）。 |
| ランドマーク | `sorta landmarks` | 常に | GPS 無しシーンの視覚的場所推定、保守的なしきい値 — 例えば屋内のランドマーク写真で GPS が無い場合に都市を補完。 |
| 顔 | `sorta faces` | **opt‑in**（`--faces`） | 顔検出（insightface）、埋め込み計算、人物クラスタリング（HDBSCAN）。最も遅い段階のため、明示的に要求しない限りスキップされます。 |
| イベント | `sorta events` | **opt‑in**（`--events`） | 時間差＋都市で写真をイベント化。日付＋都市で命名。顔検出とは独立 — どちらか、両方、どちらもオフ、を選べます。 |
| 不要写真 | `sorta junk` | 常に | 各写真を分類: `photo` / `screenshot` / `meme` / `document`（ヒューリスティック + CLIP + テキスト密度）。 |
| 類似写真ハッシュ | `sorta phash` | 常に（UI）；CLI では別コマンド（`sorta run` は呼びません — `sorta phash` を自分で実行してください） | 類似写真検出用の知覚ハッシュ。 |

**`sorta run` のフラグ**（すべて任意で、すべて「その実行だけの」上書き ——
`config.yaml` には何も書き込まれません）:

```
--faces / --no-faces       この実行で顔検出+クラスタリングを行う（既定: オフ）
--events / --no-events     この実行でイベントを構築する（既定: オフ）
--deep / --no-deep         この実行で junk 分類に深い VLM ティアを使う
                            （`uv sync --extra vlm` が必要。無ければ自動的に
                            高速 CLIP ティアへフォールバック）。既定は
                            config.yaml（naming.vlm_enabled）から。
--geo offline|online       この実行の逆ジオコーディングプロバイダ。`online` は
                            海外でより正確ですが、GPS 座標（画像は送らない）を
                            Nominatim へ送信します。既定は config.yaml
                            （geo.provider）から。
--by city|person|event     最後に振り分けのドライランプランも表示する（§9）
--dest DIR                 そのプランの出力先（省略で in-place）
```

**基本実行**（フラグ無しの `sorta run`）は意図的に高速な経路です: 都市別振り分けと
重複検出のみ、それ以外は何もしません。人物/イベント別振り分けやアルバムが本当に
必要になったら `--faces`/`--events` を有効にしてください — 後から単独で
`sorta faces`/`sorta events` を実行しても全く同じ結果になり、どちらの場合も
完全にインクリメンタルです。網羅状況はいつでも `sorta stats` で確認。

---

## 9. 振り分け：都市・人物・イベント

```bash
sorta sort --by city   --dest <dir> [--apply] [--copy|--move] [--where …] [--dedupe]
sorta sort --by person --dest <dir> [--apply] …
sorta sort --by event  --dest <dir> [--apply] …
```

- **`--by city`** → `国/都市/年/地区/…`（ローカライズ名）。
- **`--by person`** → **名前付き**人物ごとのフォルダ（先にクラスタ命名 — §11）。
- **`--by event`** → `年/イベント名/…`。
- **`--dest`** — 出力ルート。省略時は**その場（in-place）**で元フォルダを再構成
  （ドライラン・ジャーナル・undo は有効）。
- **`--copy` / `--move`** — コピー（オリジナル保持）または移動（既定）。
- **`--where`** — プランのフィルタ（繰り返し可）: `--where "country=DE" --where "year>=2020"`。
- **`--dedupe`** — 低品質の類似写真を `_Duplicates` フォルダへ。
- **`--exclude <path>`** — 整理済みサブフォルダを除外。

モードに合わない写真はレビューフォルダへ: `_Unsorted/`（場所なし／日付なし／不要写真）、
`_Documents/`（§14）。

`--apply` 無しは**ドライラン**: `report_output/`（DB 隣）に CSV と閲覧可能な HTML プランが出力
され、**何も移動されません**。

### 実例 — `--by city`

§7 の合成コレクションの続き（index/geo/junk はすでに実行済み）:

```
$ sorta sort --by city --dest sorted -c config.yaml
sort --by city (dry-run): 12 файлов -> 4 каталогов; план: …\report_output\sort_plan_city_20260721_113247.csv, …\report_output\sort_plan_city_20260721_113247.html
```

CSV プラン（1 行 1 ファイル、`target` は `--dest` からの相対パス）— ここで重要な
列だけ抜粋:

| path | country | city | target | reason |
|---|---|---|---|---|
| `Screenshots/shot_01.jpg` | | | `_未分類/ゴミ/screenshot/shot_01.jpg` | junk |
| `paris_01.jpg` | FR | Paris | `フランス/パリ/2023/paris_01.jpg` | city |
| `paris_02.jpg` | FR | Paris | `フランス/パリ/2023/paris_02.jpg` | city |
| `paris_02_edited.jpg` | FR | Paris | `フランス/パリ/2023/paris_02_edited.jpg` | city |
| `paris_03.jpg` | FR | Paris | `フランス/パリ/2023/paris_03.jpg` | city |
| `paris_04.jpg` | FR | Paris | `フランス/パリ/2023/モンマルトル/paris_04.jpg` | city — 地区（モンマルトル）が付く。日本語ロケールは地区名まで解決できる場合がある |
| `paris_05_nogps.jpg`（GPS無し） | FR | Paris | `フランス/パリ/2023/paris_05_nogps.jpg` | city — 時間的に近いパリの写真から場所を**継承** |
| `tokyo_01.jpg` | JP | Tokyo | `日本/東京都/2023/桜丘町/tokyo_01.jpg` | city — 地区（桜丘町）付き |
| `tokyo_02.jpg` | JP | Tokyo | `日本/東京都/2023/歌舞伎町/tokyo_02.jpg` | city — 別の地区（歌舞伎町） |
| `tokyo_03.jpg` | JP | Katsushika‑ku | `日本/葛飾区/2023/押上/tokyo_03.jpg` | city — 異なる GPS 座標が東京の別の区に解決された。都市が両方「東京っぽい」からといって統合されないのは正しい挙動 |

適用（`--copy` を付けてオリジナルはそのまま — 外せば移動になります）と、できあがる
ツリー:

```
$ sorta sort --by city --dest sorted_apply --copy --apply -c config.yaml
sort --by city --apply: 12 файлов -> 4 каталогов; план: …
Скопировано 12, на месте 0, ошибок 0. Откат: sorta undo

$ find sorted_apply -type f
sorted_apply/フランス/パリ/2023/paris_01.jpg
sorted_apply/フランス/パリ/2023/paris_02.jpg
sorted_apply/フランス/パリ/2023/paris_02_edited.jpg
sorted_apply/フランス/パリ/2023/paris_03.jpg
sorted_apply/フランス/パリ/2023/モンマルトル/paris_04.jpg
sorted_apply/フランス/パリ/2023/paris_05_nogps.jpg
sorted_apply/フランス/パリ/2023/person_a_1.jpg
sorted_apply/日本/葛飾区/2023/押上/tokyo_03.jpg
sorted_apply/日本/東京都/2023/桜丘町/person_a_2.jpg
sorted_apply/日本/東京都/2023/桜丘町/tokyo_01.jpg
sorted_apply/日本/東京都/2023/歌舞伎町/tokyo_02.jpg
sorted_apply/_未分類/ゴミ/screenshot/shot_01.jpg

$ sorta undo -c config.yaml
Откат батча 2: возвращено 12, отсутствовало 0, ошибок 0

$ find sorted_apply -type f
（何も出ない — undo が全コピーを削除）
```

同じ写真を `language: en`/`ru` で振り分けると `France/Paris/2023/…`、
`Франция/Париж/2023/…` になり、東京側の地区サブフォルダは付きません — バグでは
ありません。同梱の GeoNames データには日本語ローカライズされた地区名があります
が、`en`/`ru` には存在せず、`naming.drop_unlocalized_district`（既定オン）が、
ローカライズできない言語では生の翻字コードを表示する代わりに地区セグメントを
省略するためです。

**`--where` によるフィルタ:**

```
$ sorta sort --by city --dest sorted_fr --where "country=FR" -c config.yaml
sort --by city (dry-run): 7 файлов -> 1 каталогов; план: …
```

フランスに解決された 7 ファイルだけがプランに含まれます。それ以外は
（`_未分類` にも送られず）プランから完全に除外されます。

### 実例 — `--by event`

```
$ sorta sort --by event --dest sorted_event -c config.yaml
sort --by event (dry-run): 12 файлов -> 3 каталогов; план: …
```

| path | event | target |
|---|---|---|
| `paris_01.jpg` … `person_a_1.jpg`（7 ファイル） | `2023-06-10..06-11 Paris` | `2023/2023-06-10..06-11 Paris/<名前>.jpg` |
| `tokyo_01.jpg`、`tokyo_02.jpg`、`tokyo_03.jpg`、`person_a_2.jpg`（4 ファイル） | *（なし — `events.min_event_size` 未満）* | `2023/11/<名前>.jpg` — `no_event` フォールバック、代わりに年/月でグループ化 |
| `shot_01.jpg` | | `_未分類/ゴミ/screenshot/shot_01.jpg` — モードに関わらず junk が優先 |

これは §7/§12 と同じ `min_event_size` のしきい値の実例です: 東京の 1 日には実際の
GPS、実際のタイムスタンプ、実際の場所がありましたが、単独のイベントとして名前が
付くのに十分な**ファイル数**だけが足りませんでした。

### 実例 — `--by person`

person モードにはまず**名前付きの顔クラスタ**（§11）が必要で、そのためには実在の
写真から `sorta faces` が実際に顔を見つける必要があります — これは、私たちの合成
プレースホルダー画像では誠実に実演できません（§7 の注記と §11 の補足を参照）。実在
コレクションでいくつかクラスタに名前を付けた後は、次のような形になります:

```bash
sorta sort --by person --dest /path/to/sorted --apply
```

これにより、その人物が（あるいは主たる人物 — §21 の `sort.multi_person` 参照）
名前付きの顔である写真すべてが `<dest>/<人物名>/<file>.jpg` に振り分けられます。
名前付きの人物がいない写真にもどこかへの行き先が必要なので、`_未分類/` に
フォールバックします。junk/スクリーンショットの振り分けや
`--where`/`--copy`/`--move`/`--apply` は、上の city/event の例と全く同じように
動作します。

---

## 10. 重複

- **完全重複**（バイト一致）は `index` 中に検出。正規ファイルのみ振り分けられ、他は
  その場に残ります。
- **類似写真**（見た目が近く、サイズ／名前が異なる）は知覚ハッシュで検出（`sorta
  phash` → `sorta dupes --near` または UI の **重複** タブ）。

UI の **重複** タブ: 各グループにおすすめの残す 1 枚（★）。異なる所だけラジオを変更し、
*「このグループは削除しない」*でスキップ、最後に一度だけ **すべての選択を保存**
（グループごとのクリック不要）。次の振り分け／コピー時、残さない写真は `_delete`
フォルダへ（復元可能）。または各写真の **削除** ／ **重複を削除** で OS のゴミ箱へ即時。

§7 の合成コレクションでの実際の出力（`sorta phash` 実行後）:

```
$ sorta dupes -c config.yaml
paris_01_copy.jpg
  -> дубликат paris_01.jpg

Всего: 1

$ sorta dupes --near -c config.yaml
Группа из 2 похожих:
  paris_02.jpg  (7424 байт)
  paris_02_edited.jpg  (5908 байт)
Группа из 2 похожих:
  person_a_1.jpg  (14742 байт)
  person_a_2.jpg  (14742 байт)

Групп: 2 (порог Хэмминга: 5)
```

`paris_02_edited.jpg` は `paris_02.jpg` を実際に再圧縮・縮小したコピーで、
知覚ハッシュ本来の「同じ写真を編集・再エクスポートした」ケースそのものです。
2 つ目のグループは、誤検出だが示唆に富む例です: 2 枚の顔プレースホルダー画像は
（同じ手順で生成したため）ピクセル単位で同一なので、pHash は正しく両者を類似写真
と判定しますが、`sorta faces` はどちらも顔なしの無関係なファイルとして扱います。
実在のコレクションでは、同一人物の異なる 2 枚の写真は通常「類似写真」には**なりま
せん** — pHash は画像全体の類似度を比較するのであって、人物の同一性ではありません。

---

## 11. 人物と顔クラスタ

顔検出は**クラスタ**（同一の顔のまとまり）を作ります。人物振り分けを意味あるものに
するため、クラスタに名前を付けます:

- **UI → 人物 タブ:** クラスタにサンプル顔が表示。名前を入力して **名前を付ける**。
  2 つのクラスタを選び、同一人物なら **統合**。
- **CLI:** `sorta faces label <cluster_id> "母"`、`sorta faces merge <src> <dst>`、
  `sorta faces sheet <cluster_id> out.html`（識別用コンタクトシート）。

命名後、`sorta sort --by person`（または人物**アルバム**、§13）が名前を使います。

`sorta faces` の実行には `sorta run` の `--faces` フラグ、または UI の「顔を検出」
チェックボックスが必要です（§8）— 基本パイプラインでは実行されません。§7 の合成
コレクションでの実際の出力:

```
$ sorta faces -c config.yaml
Детекция: 12 файлов, 0 лиц, 12 без лиц, 0 ошибок
Кластеры: 0 (лиц в кластерах: 0, шум: 0, имён сохранено: 0)
```

本当にゼロです — buffalo_l は実在の写真で訓練されており、私たちの合成プレース
ホルダー画像（フラットなベクター図形で、実際の顔のテクスチャではない）には正しく
反応しません。これは想定どおりであり、Sorta やこのガイドのバグではありません。
実際の写真コレクションに `sorta faces` を向ければ、実在の顔を検出します。検出でき
れば（この合成実行ではなく実際の実行なら、たとえば `Детекция: 340 файлов, 512
лиц, 8 без лиц, 0 ошибок` / `Кластеры: 6 (лиц в кластерах: 480, шум: 32, имён
сохранено: 0)` のような出力になります）、命名と振り分けは上と全く同じコマンドです
— `sorta faces label 3 "母"` はクラスタ `3` に名前を付け、続けて
`sorta sort --by person --dest … --apply` がその人物の写真を `<dest>/母/` に
振り分けます。

---

## 12. イベント

イベントは時間差と都市で写真をまとめます。`sorta events` が（再）構築:

- 小さなまとまり（`events.min_event_size` 未満）はイベント化しません。
- 同一都市のセッションは `events.trip_merge_gap_hours` 以内で 1 つの旅行に統合。
- 名前 = 日付範囲 + ローカライズ都市（例: `2023-11-29..12-02 Sochi`）。

手動操作:
- `sorta events add "会議" 2025-05-21 2025-05-23` — 日付範囲の手動イベント（再計算後も
  維持）。
- `sorta events rename <event_id> "IEEE 会議 東京"` — 手動の名前。

`sorta events` の実行には `sorta run` の `--events` フラグ、または UI の「イベント
を検出」チェックボックスが必要です（§8）。§7 の合成コレクションでの実際の出力
（パリの 7 ファイルは既定の `min_event_size`＝5 を超え、東京の 4 ファイルの日は
超えません）:

```
$ sorta events -c config.yaml
События: 1 авто (7 файлов, имён сохранено: 0), 0 ручных (0 файлов)
```

---

## 13. アルバム

**アルバム**は特定のスライス — 1 人物（任意でフィルタ）または 1 イベント — を、都市の
正規構成を崩さずに専用の名前付きフォルダへ集めます。

```bash
# 「母」の全写真をハードリンク（既定）で。まずプレビュー、次に適用:
sorta album person "母" --dest /path/to/albums
sorta album person "母" --dest /path/to/albums --apply

# 「母」だがバルセロナのみ:
sorta album person "母" --where "city=Barcelona" --dest /path/to/albums --apply

# 特定イベントを独自フォルダ名・コピーで:
sorta album event "2025-05-21..05-23 Tokyo" --dest /path/to/albums \
      --name "IEEE 会議 東京" --copy --apply
```

- 既定モードは **link**（ハードリンク、追加容量ほぼゼロ。1 枚が複数アルバム*と*都市構成
  の両方に存在可）。
- **`--copy`** は独立コピー。**`--move`** は*ファイルを共通プールから取り出します*
  （警告表示）。**2 人以上の名前付き人物**が写る写真は 1 つのアルバムへ move 不可
  （曖昧）— ブロックされます。link/copy を使ってください。
- UI では 人物／イベント カードの **フォルダに集める** ボタン。

実際の出力 — §12 のパリのイベントをコピーでアルバムに集める例:

```
$ sorta album event "2023-06-10..06-11 Paris" --dest albums --copy --apply -c config.yaml
album event '2023-06-10..06-11 Paris' --apply [copy]: 7 файлов -> …\albums\2023-06-10..06-11 Paris
Альбом «2023-06-10..06-11 Paris»: выгружено 7, ошибок 0. Откат: sorta undo

$ find albums -type f
albums/2023-06-10..06-11 Paris/paris_01.jpg
albums/2023-06-10..06-11 Paris/paris_02.jpg
albums/2023-06-10..06-11 Paris/paris_02_edited.jpg
albums/2023-06-10..06-11 Paris/paris_03.jpg
albums/2023-06-10..06-11 Paris/paris_04.jpg
albums/2023-06-10..06-11 Paris/paris_05_nogps.jpg
albums/2023-06-10..06-11 Paris/person_a_1.jpg
```

`--copy` を渡したので、これらは独立したファイルです — `sorta undo` はアルバムの
コピーだけを削除し、オリジナルには触れません（§15 参照）。

---

## 14. 不要写真・スクショ・書類

`sorta junk` は各写真を分類し、都市／人物／イベントのフォルダから「思い出でないもの」を
除外します:

- **`screenshot`**、**`meme`** → `_Unsorted/junk/…`。`Screenshots/` フォルダ内のファイル
  はフォルダ名でも検出。
- **`document`**（パスポート、レシート、書式、診断書…）→ `_Documents/` — 不要写真では
  なく**レビューフォルダ**。検出は CLIP と**テキスト密度**シグナルを併用（書類は
  テキストが密。ビーチや商品写真は密でない）。

`_Documents/` は意図的に**多めに集めます**（本物の写真は簡単に取り出せる。本物の書類が
都市の思い出に紛れる方が問題）。手動で確認してください。注: 「販売用商品」写真を書類から
自動で分ける精度は既知の制約で、`_Documents/` に混在し得ます。

> **プライバシー:** 書類には個人情報が含まれ得ます。Sorta は**ローカル**で処理し、外部へ
> 送信しません（オンラインプロバイダーを有効化しない限り）。§15 参照。

§7 の合成コレクションでの実際の出力 — `Screenshots/` に保存した 1 枚だけがフォルダ
名のヒューリスティックで検出され、残りは通常の写真として分類されます:

```
$ sorta junk -c config.yaml
Классификация: 12/12 обработано (photo: 11, screenshot: 1)
```

---

## 15. 安全性・取り消し・プライバシー

- **既定でドライラン** — `--apply` まで何も動きません。
- **移動ジャーナル** — 各操作は実行*前*に記録。
- **取り消し** — `sorta undo` が直前バッチを戻します（`--batch <id>` で特定）。copy/link
  バッチの undo はコピー／リンクを削除し、オリジナルは触りません。
- **ハッシュ照合・上書きなし** — 移動前に blake3 を照合、名前衝突は `_1`、`_2`。
- **copy/link ならオリジナル無傷。** move ではファイルは移動しますが、内容と EXIF は不変。
- **既定でローカル。** 顔／シーン／テキストモデルは自分のマシンで動作。オンライン
  プロバイダーは `config.yaml` で**オプトイン**です: `geo.provider: online`
  （Nominatim）は GPS 座標のみを送信し画像は送りません。`naming.provider: claude`
  はイベントごとのサンプル写真を数枚 Claude API へ送信します（実際に写真の内容が
  マシンの外に出る唯一の機能）。各プロバイダが正確に何を送信するかは
  [SECURITY.md](../../SECURITY.md) を参照してください。プライバシー最大化には
  両方とも無効のままに。
- Web UI は `127.0.0.1` のみ待受。

---

## 16. コマンド一覧

```
sorta index [DIR]                 sources（または DIR）をスキャン → メタデータ・ハッシュ・完全重複
sorta index --refresh-exif        インデックス済みファイルのメタデータを読み直す（§17）
sorta run [--src DIR] [--faces] [--events] [--deep/--no-deep] [--geo offline|online]
          [--by city|person|event] [--dest DIR]
                                  基本パイプライン（index→geo→landmarks→junk）;
                                  --src はこの実行だけ config の sources を上書き;
                                  --faces/--events は遅い段階を opt-in（既定オフ、
                                  互いに独立）; --deep/--geo はこの実行だけ
                                  config.yaml を上書き; --by 指定時は末尾に
                                  ドライランプランを表示
sorta geo                         場所を解決（GPS + セッション推定）
sorta landmarks                   GPS 無しシーンの視覚的場所推定（保守的）
sorta faces                       顔検出 + 人物クラスタリング
sorta faces label <cluster> <name>    クラスタに名前
sorta faces merge <src> <dst>          2 クラスタを統合（同一人物）
sorta faces sheet <cluster> <out.html> 識別用コンタクトシート
sorta events                      イベントを（再）構築
sorta events add <name> <from> <to>    日付範囲の手動イベント
sorta events rename <id> <name>        手動のイベント名
sorta junk                        photo/screenshot/meme/document を分類
sorta phash                       知覚ハッシュ（類似写真用）
sorta stats                       インデックス網羅状況（GPS、日付ソース、重複）
sorta dupes [--near]              完全／類似 重複の一覧
sorta sort --by MODE [--dest DIR] [--apply] [--copy|--move]
           [--where …] [--dedupe] [--delete-worse-dupes] [--exclude PATH] [--thumbnails]
                                  振り分けのプラン／適用（--apply 無しはドライラン）
sorta album person|event <selector> --dest DIR [--copy|--move] [--where …] [--name N] [--apply]
                                  スライスを名前付きフォルダに集める（既定はハードリンク）
sorta undo [--batch ID]           直前（または指定）バッチを取り消し
sorta reset [--yes]               インデックス（DB）を消去してやり直す — 写真や
                                  振り分け済みフォルダには触れません（人物/イベント名
                                  と重複判定は失われます）
sorta ui [--port 8756]            ローカル Web アプリ（処理／都市／重複／人物／イベント／移動）
sorta doctor                      環境診断: torch/onnxruntime、GPU、地理データ、
                                  ログのパス、プレビューキャッシュのパス（§3.5）
sorta cache [--clear]             プレビューキャッシュ: サイズ表示、または削除（§18）
```

各コマンドは `-c/--config <path>`（既定 `config.yaml`）を受け付けます — ただし
`sorta doctor` だけは設定ファイルを一切読みません。

---

## 17. メンテナンスと診断

パイプラインには含まれないものの、日常的に使う 3 つのコマンドです。

### `sorta doctor`

§3.5 で説明したインストール／環境の診断です: torch と onnxruntime のデバイス、
同梱の地理データベース、実行ログのパス、プレビューキャッシュのパス。インストール
やプロファイル変更のたびに、また「妙に遅い」と感じたときは真っ先に実行してくだ
さい。

### `sorta index --refresh-exif`

**すでにインデックスにあるファイル**のメタデータを、再インデックスせずに読み直し
ます。要約行の形は次のとおりです（`<N>` は実行ごとのカウンタ）:

```
$ sorta index --refresh-exif -c config.yaml
Перечитано: <N> файлов, обновлено <N>; вернулось координат: <N>, дат съёмки: <N>; без EXIF: <N>, ошибок: <N>
Появились новые координаты — перезапустите: sorta geo (и sorta events)
```

- **なぜ必要か。** 通常の `sorta index` はこれらのファイルを意図的にスキップします:
  パス + サイズ + mtime を比較しており、そのどれも変わっていないからです。変わった
  のが*読み取り側*（メタデータ抽出を修正した Sorta の更新など）である場合、通常の
  実行では二度と再訪されません。
- **いつ実行するか。** EXIF／メタデータの読み取りに触れた更新のあと、あるいは
  `sorta stats` の GPS・EXIF 日付の件数が想定より少ないときです。
- **何が書き換わるか。** GPS、カメラのメーカー／機種、寸法、撮影日時、向きだけです。
  ハッシュ、pHash、重複マークには**触れません** — ファイルの内容は変わっておらず、
  変わったのは「そこから読み取れた内容」だけだからです。ファイル全体の読み込みも
  再ハッシュも行いません。
- **その後に。** 座標が復活したということは、場所を解決し直す必要があるという
  ことです: `sorta geo` を、イベントを使っているなら `sorta events` も実行して
  ください。実際に座標が戻った場合、コマンド自身がその旨を表示します。

### `sorta cache`

プレビューキャッシュ（§18）の状況を表示、または削除します:

```
$ sorta cache -c config.yaml
Кэш превью: C:\Users\you\AppData\Local\sorta\previews
  файлов: 34887, размер: 12.60 ГБ

$ sorta cache --clear -c config.yaml
Кэш превью удалён: C:\Users\you\AppData\Local\sorta\previews
```

キャッシュはいつ削除しても安全です: 遅延生成なので、そのフレームを次に必要とした
段階が 1 枚ずつ作り直します。コストは、そのフレームをオリジナルからもう一度
デコードすることだけです。

---

## 18. プレビューキャッシュ

**何をするものか。** 各フレームのデコードは**一度だけ**。その結果（長辺 1536 px に
縮小した JPEG）が共有キャッシュディレクトリに書かれ、以降の段階（pHash、CLIP、OCR、
深い VLM ティア、UI のサムネイル）はオリジナルを再デコードせずにその小さなコピーを
読みます。

**なぜ。** これらの段階が実際に時間を使っているのはデコードだからです。HEIC の
フルデコードは約 470 ms、同じフレームをプレビューキャッシュから読むと 1 桁 ms
（約 8 ms）です。以前は段階が順番に走るため、同じ写真が 1 回の実行で 3〜5 回
デコードされていました。

**保存場所。**

| OS | パス |
|---|---|
| Windows | `%LOCALAPPDATA%\sorta\previews` |
| Linux / macOS | `~/.cache/sorta/previews` |

**容量の目安: 1 枚あたり約 150 KB、合計はギガバイト単位です。** 150 KB は 1536 px
q88 の JPEG としての設計値で、精細なフレームはもっと大きくなります。実測では
34,887 件のプレビューで **12.60 GB**（1 件あたり約 360 KB）でした。数万枚規模の
コレクションでは数ギガバイトのディスクを使うということであり、誤差ではありません。
これはユーザー領域のキャッシュで、写真コレクションの中には決して書き込まれません。
現在のサイズはいつでも `sorta cache` で確認できます（§17）。

**小さい画像はキャッシュされません。** オリジナルが既に `preview_max_edge` 以下なら、
プレビューは節約ではなく単なるコピーになるためスキップされます — スクリーンショット
ばかりのコレクションはここで容量を使いません。

**設定** — `config.yaml` の `imaging:` セクション:

```yaml
imaging:
  preview_cache: true     # false → 各段階が毎回オリジナルをデコードし直す
  # preview_dir: D:/sorta-previews   # 既定は上の表のとおり
  preview_max_edge: 1536  # キャッシュする JPEG の長辺
  preview_quality: 88     # キャッシュする JPEG の品質
```

各キーには同じ形の環境変数（`SORTA_PREVIEW_CACHE`、`SORTA_PREVIEW_DIR`、
`SORTA_PREVIEW_MAX_EDGE`、`SORTA_PREVIEW_QUALITY`）があり、両方指定されている場合は
**環境変数が優先**されます。

---

## 19. 実行ログ

長い実行の記録はコンソールには残りません — とくに `sorta ui` は誰も見ていない
バックグラウンドのウィンドウで何時間も動き続けます。そのため、どのコマンドも
ログファイルを書き出します。

| OS | パス |
|---|---|
| Windows | `%LOCALAPPDATA%\sorta\logs\sorta.log` |
| Linux / macOS | `~/.cache/sorta/logs/sorta.log` |

ローテーションは **5 MB × 5 世代**（`sorta.log.1` … `sorta.log.5`）なので、無制限に
増えることはありません。パイプラインの実行（`sorta run`）と `sorta ui` の起動時には、
さらに環境ヘッダ（Sorta と Python のバージョン、プラットフォーム、`sorta` パッケージ
の読み込み元、GPU の状態、地理データ、作業ディレクトリ）が記録されます —
`sorta doctor` が表示するのと同じ事実を、実行時点のスナップショットとして残した
ものです。

**段階ごとの所要時間。** パイプライン実行（`sorta run` または UI の「処理」ボタン）の
各段階は `stage=<名前> … elapsed=<秒>` という安定した形の行を出力するため、文章を
解析しなくても実行のプロファイルが読めます。2026‑07‑25 の実際の実行のプロファイル:

```
2026-07-25T23:37:10.345 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=index started
2026-07-25T23:37:43.854 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=index elapsed=33.509
2026-07-25T23:37:43.855 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=geo started
2026-07-25T23:37:47.727 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=geo elapsed=3.872
2026-07-25T23:37:47.727 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=landmarks started
2026-07-25T23:41:34.228 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=landmarks elapsed=226.501
2026-07-25T23:41:34.229 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=junk started
2026-07-26T00:10:34.736 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=junk elapsed=1740.508
2026-07-26T00:10:34.736 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=phash started
2026-07-26T00:26:09.900 INFO sorta.runlog [Thread-23 (_run_pipeline)] stage=phash elapsed=935.164
```

そのまま読み取れます: `junk` が 29 分、`phash` が 15.6 分で、この 2 つだけで 49 分の
実行の 91 %。`index`（33 秒）と `geo`（4 秒）は誤差です。`gpu` エクストラでのインストール、
`naming.clip.decode_workers`、`naming.ocr_workers`（§21）を投入すべき先はそこだと
分かりますし、「固まった」のか「遅い段階の途中」なのかもこれで区別できます。

同じ行の別の形: `stage=<名前> failed elapsed=…`（直後にトレースバック）と、
キャンセルされた実行の `stage=<名前> interrupted (…) elapsed=…` — UI の「キャンセル」
はエラーではありません。件数を報告する段階では ` processed=<n> rate=<n>/s` も
付きます。

**上書き設定。**

- `SORTA_LOG_FILE=<パス>` — ログの出力先を変更。
- `SORTA_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` — **ファイル**側のレベル（既定 `INFO`）。

ファイル側のレベルはコンソール側とは独立です: `config.yaml` の `log_level` が
制御するのはコンソール出力だけ（既定 `WARNING` — 警告とエラー）で、ファイルには
段階ごとの所要時間を含む `INFO` の詳細が残ります。コンソールを静かにしても、
ログファイルは静かになりません。

---

## 20. モデルのオフライン動作

ML モデル（CLIP、easyocr、使用していれば VLM ティア）のダウンロードは**一度だけ**
です。それ以降、これらのためにネットワークは一切不要になります — 「既定でローカル」
が実際に意味するのはこういうことです。

切り替えは自動です: Hugging Face のキャッシュにダウンロード済みモデルが 1 つでも
あれば、Sorta は各コマンドをオフラインモード（`HF_HUB_OFFLINE` /
`TRANSFORMERS_OFFLINE`）で開始し、すでにディスクにある重みのリビジョン確認のために
Hub へ問い合わせる段階は 1 つもありません。キャッシュが空の新しいマシンはそのまま
なので、最初のダウンロードは従来どおり行えます。

- `SORTA_ALLOW_MODEL_DOWNLOAD=1` で自動切り替えを**無効化**できます — 他のモデルが
  すでにキャッシュされているマシンで*新しい*モデルを取得したいとき（VLM ティアを
  初めて有効にする場合など）に使います。
- `HF_HUB_OFFLINE` や `TRANSFORMERS_OFFLINE` を自分で設定している場合、Sorta が
  その値を上書きすることはありません。

なお、顔検出（insightface/`buffalo_l`）は `~/.insightface/models` に独自の
キャッシュを持ち、これらの環境変数の影響は受けません。

---

## 21. 設定リファレンス

`config.yaml` の主なセクション（完全なテンプレートは `config.example.yaml`）:

```yaml
sources: ["D:/Photos"]         # スキャン対象フォルダ（再帰）
database: "sorta.db"           # SQLite インデックスのパス
language: ja                   # ru | en | ja — フォルダと UI の言語

index:
  min_file_size_kb: 5          # 極小ファイルを無視
  workers: 8                   # 並列ハッシュ計算
  exif_workers: 8              # 並列 exiftool セッション（0/未指定 = 自動 min(8, コア数)）
  skip_dirs: [".thumbnails", "@eaDir", "$RECYCLE.BIN", "System Volume Information"]

geo:
  provider: offline            # offline（同梱 GeoNames）| online（Nominatim/OSM）
  session_gap_hours: 6         # GPS 推定セッションを分ける時間差
  nominatim_url: "https://nominatim.openstreetmap.org"   # provider: online のみ
  nominatim_user_agent: "sorta-photo-organizer"          # OSM ポリシー上必須

events:
  gap_hours: 6                 # 新セッションを開始する時間差
  trip_merge_gap_hours: 48     # 同一都市のセッションを旅行に統合する上限
  min_event_size: 5            # これ未満のまとまりはイベント化しない

sort:
  multi_person: primary        # 複数人物写真 → 最大の顔の人物へ
  exclude_dirs: []             # 振り分けでスキップするサブフォルダ
  album_dir: null              # アルバムのルート（既定: DB 隣の _Albums）
  report_dir: null             # sort プラン(CSV/HTML)の出力先（既定: DB 隣の report_output/）

faces:
  min_face_px: 40              # これより小さい顔は無視
  det_threshold: 0.7           # 検出器の信頼度
  min_cluster_size: 5          # クラスタ最小顔数（HDBSCAN）
  max_distance: 0.5            # コサイン類似度しきい値

naming:
  landmark_threshold: 0.85     # 視覚的場所の CLIP しきい値（保守的）
  junk_threshold: 0.85         # screenshot/meme の CLIP しきい値
  document_threshold: 0.9      # 書類の CLIP しきい値
  text_frac_document: 0.15     # テキスト面積比がこれ以上なら写真 → 書類
  text_rescue_docscore_min: 0.3  # doc-score がこれ以上の写真のみ OCR で再判定
  ocr_workers: 4               # 並列 OCR 検出器（既定 min(4, コア数)）
  vlm_enabled: false           # 深い VLM 分類ティア（`vlm` エクストラが必要）;
                               #   `--deep` / UI の「詳細解析」チェックボックスと同じ
  clip:
    batch_size: 16             # GPU での CLIP フォワードのバッチサイズ
    decode_workers: 0          # CLIP 用デコードスレッド; 0 = 自動 min(コア数, 16)

imaging:                       # プレビューキャッシュ — §18 参照
  preview_cache: true
  preview_max_edge: 1536
  preview_quality: 88

log_level: WARNING             # コンソールのみ; 実行ログは INFO のまま（§19）
```

### スループット関連の設定

重い段階の速度を決めるつまみは 4 つです。いずれも任意で、既定値は控えめな
ハードウェアでも問題が出ないように選ばれています。詳しい解説と実測値は
`config.example.yaml` にあります。

| 設定 | 何をするか | いつ触るか |
|---|---|---|
| `index.exif_workers` | `index` 段階での並列 `exiftool` セッション数。exiftool は別プロセスなのでほぼ線形にスケールします（実測: 1 → 8 セッションで 1 ファイルあたり 11.8 → 2.0 ms）。既定は自動 `min(8, コア数)`。 | ログで `stage=index` が支配的なら、コア数の多いマシンで増やす。 |
| `naming.ocr_workers` | `junk` 段階の並列 OCR 検出器数。各ワーカーが**自分の**モデルを VRAM に持つため、既定はあえて控えめな `min(4, コア数)`。 | VRAM に余裕があるときだけ増やす。OCR でメモリが足りなくなるなら減らす。 |
| `naming.clip.batch_size` | GPU 上の CLIP フォワードのバッチサイズ。CLIP 経路はデコード律速なので、ほとんど効きません。既定 16。 | 変更が必要になることはほぼありません。 |
| `naming.clip.decode_workers` | CLIP に供給するデコードスレッド数 — `junk`/`landmarks` の速度を左右する本命のつまみ。既定は自動 `min(コア数, 16)`。 | `stage=junk`/`stage=landmarks` が支配的で GPU が遊んでいるなら増やす。 |

---

## 21a. フォルダーの除外と手動での修正

目的の異なる二つの機能です。取り違えやすいため、用語は意図的に分けています。

- **「スキャンしない」** — ファイルはインデックスに一切入りません。
- **「配置しない」** — インデックスには入りますが、元の場所に留まります。

### スキャン前にソースフォルダーを除外する

ソースフォルダーは、走査が到達する*前*に除外できます。作業量を実際に減らせるのは
この方法だけです。除外されたサブツリーは `stat` もハッシュ計算も、以降のどの段階も
消費しません。実際の 400 GB のコレクションでは、`Movies` フォルダーを一つ除外する
だけで 847 ファイル・54 GB が全段階から外れました。

ウェブアプリでは最初のタブにソースのフォルダーツリーが表示されます。各フォルダーは
☐ 処理する・◐ 振り分けない・☒ スキャンしない の三状態を持ち、マークをクリックすると
順に切り替わります。親の状態はサブツリー全体に適用されます。コマンドラインからは
次のとおりです。

```bash
sorta index --exclude-dir Movies --exclude-dir Screenshots
```

どちらも同じファイル（データベースの隣の `excludes.yaml`、または
`index.excludes_file` が指す場所）に書き込むため、一方で除外したフォルダーは
もう一方でも除外されたままです。ファイルは**ソースのルートごと**に、さらにルート内
では除外の意味ごとに整理されます。

```yaml
"D:/Photos":
  skip_scan:      # スキャンしない: ファイルはインデックスに入りません
    - Movies
    - DCIM3/Screenshots
  skip_layout:    # 振り分けない: インデックスに入り、その場に残ります
    - foto/Greece
    - foto/France
"E:/Archive":
  skip_scan:
    - temp
```

フォルダーはどちらか一方の区分にのみ入ります。一方を選ぶともう一方は解除されます。
旧形式のファイル（ルート直下の単純なリスト）は `skip_scan` として読み込まれるため、
書き換えの必要はありません。

そのため、ソースを切り替えても古い除外設定をどうするか判断する必要はありません。
ルートごとに独自の設定を持ち、戻れば復元されます。なお、新たに除外されたパスの下に
すでにインデックス済みのファイルがある場合、次回の `index` 実行時に
**インデックスから削除**されます。「スキャンしない」と「インデックスにある」は
両立しません。移動履歴には手を触れません。

### 個々のファイルの修正

自動的な規則は、あなたに見えているものを見ていません。ウェブアプリでは任意の
ファイル（または選択範囲）に対して次の操作ができます。

- **配置から除外** — コピーも移動もされず、計画にも現れません。
- **移動先の変更** — 現在の配置内の任意のフォルダーへ。

どちらも自動的な規則（重複の判定、分類の結果、位置情報）より優先されます。設定は
インデックスに保存され、ゼロからの再インデックスで消えます。顔クラスターの名前や
イベント名と同じ扱いです。

### ファイルの元の場所

各行には、そのファイルが元々あったフォルダーが表示され、完全なパスはツールチップに
出ます。誤った推測を見分けるのに最も速い方法であることが多く、カレリアの公園の名前
が付いたフォルダーにあると分かれば、コロッセオという判定が誤りだと一目で分かります。

---

## 21b. 信頼できる日付のないファイル

日付を特定できなかった写真は、ダウンロードされた画像と同じ場所には入りません。
カメラの痕跡（メーカー、機種、GPS のいずれか）があるかどうかで分けられます。

- 日付を失ったカメラ写真 — 「日付なし」フォルダー。
- カメラの痕跡が一切ない — **「ダウンロード」**フォルダー。

検証用コレクションでは、この境界はほぼ完全に一致しました。日付のない 1 059 件の
うち 1 057 件にはカメラの痕跡がなく、`10083666931142353280.JPG` のような名前の
メッセンジャーのキャッシュでした。本物の写真はちょうど 2 件です。フォルダー名は
判定のように聞こえないよう意図的に選んでいます。転送された画像も目を通す価値がある
ことが多いためです。


---

## 22. トラブルシューティング

- **GPU 搭載機なのに `sorta doctor` が `torch: 2.13.0+cpu` と表示する** — エクストラ
  がインストールコマンドに届いていません。`uv tool install` に `--extra` フラグは
  無く、エクストラは引用符付きの指定の内側に書きます:
  `uv tool install --force "C:\path\to\sorta[gpu]"`（§3.3）。
- **torch は CUDA を認識しているのに `sorta doctor` に `CUDAExecutionProvider` が無い**
  — `insightface` が連れてくる素の `onnxruntime` が `onnxruntime-gpu` を上書きし、
  顔検出が CPU で動いています。説明と回避策は §3.6。
- **コードを変更しても反映されない** — editable でない `uv tool install` は
  インストール時点のスナップショットです。`-e` を付けて入れ直す（§3.3）か、
  プロジェクト venv を使ってください（§3.4）。
- **`uv sync`（extra 無し）で `sorta faces`/`sorta junk` が壊れる、または一貫しない**
  — 想定どおりです。必ず明示的なプロファイルでインストールしてください:
  `uv sync --extra cpu --extra dev` または `uv sync --extra gpu --extra dev`
  （§2/§3）。`cpu`/`gpu` は互いに排他的です。後でハードウェアを変える場合は、
  別の extra で `uv sync` をやり直すだけです。
- **`No module named ruff` / dev ツールが無い** — `uv sync` に `--extra dev` を
  追加してください（上記の cpu/gpu プロファイルとは別物です）。
- **HEIC/RAW の日付・プレビュー・動画メタデータが無い** — `exiftool` を導入
  （§3）。これらの形式には必須で、Pillow は JPEG/PNG/TIFF/WEBP しかカバーしま
  せん。
- **GPU プロファイルで顔／CLIP が非常に遅い** — 実際に `uv sync --extra gpu`
  （`cpu` ではなく）を実行したか、ドライバが CUDA 13 に対応しているか確認して
  ください。`sorta faces`/`sorta junk` は出力の先頭付近で、onnxruntime がどの
  execution provider を選んだか（`CUDAExecutionProvider` か
  `CPUExecutionProvider` か）を表示します。
- **`--extra gpu` を入れたのに分類/顔検出が遅い** — おそらく `sorta` を
  素の `uv run sorta …` 経由で実行しています。`uv run` は実行のたびに
  環境を `pyproject.toml` の基本依存関係へ再同期し、そのコマンドで毎回
  `--extra gpu` を繰り返し指定しない限り GPU 版 torch を CPU に戻して
  しまいます(§3 参照)。`uv tool install` でインストールしたバイナリ
  (`uv tool install ".[gpu]"` の後は素の `sorta …`)か、アクティベートした
  venv を使ってください — どちらも実行のたびに再同期されません。GPU が
  実際に使われているかの確認: `python -c "import torch;
  print(torch.cuda.is_available())"` が `True` を出すはずです。ハードウェア
  プロファイルの変更 = 別の extra で再インストール、という §3 と同じ手順です。
- **GPU プロファイルで意図的に CPU を強制する**（デバッグ用、または GPU が他の
  用途で使用中など）— コマンドに対して `CUDA_VISIBLE_DEVICES=`（空）を設定して
  ください。torch も onnxruntime もこれを尊重して CPU にフォールバックします:
  ```bash
  CUDA_VISIBLE_DEVICES= sorta faces          # bash/macOS/Linux
  ```
  ```powershell
  $env:CUDA_VISIBLE_DEVICES=''; sorta faces  # PowerShell
  ```
- **`buffalo_l` が毎回再ダウンロードされる** — モデルキャッシュ
  （`~/.insightface/models/buffalo_l`）が削除されたか書き込み不可になっています。
  このパス（または実際にモデルが置かれている場所へのシンボリックリンク/
  junction）が実行間で維持されるようにしてください。
- **`database is locked`** — 別の Sorta プロセスが書込み中（例: パイプライン実行）。完了
  を待ち、書込みを 2 つ同時に走らせないこと。
- **実行中にディスクの空きがギガバイト単位で減った** — おそらくプレビュー
  キャッシュ（§18）です: 1 枚あたり約 150 KB 以上。`sorta cache` で確認し、
  `sorta cache --clear` で削除、`imaging.preview_dir` で移動、
  `imaging.preview_cache: false` で無効化できます（無効化すると各段階で毎回
  デコードし直すコストがかかります）。
- **他のモデルが既にあるマシンで新しいモデルがダウンロードされない** — 想定どおり
  です: キャッシュが空でなくなった時点で Sorta は Hugging Face をオフラインに
  切り替えます（§20）。`SORTA_ALLOW_MODEL_DOWNLOAD=1` を付けて一度実行すれば、
  新しい重みを取得できます。
- **非 ASCII 名（例: キリル文字）のフォルダが OCR でスキップされたように見える** — 修正
  済み。画像は Unicode 安全なパスでデコードされます。最新版へ更新を。
- **`language: en`/`ru` でも CLI のコンソールメッセージがロシア語で表示される** —
  想定どおりで、バグではありません: `language` はフォルダ名と Web UI を制御する
  もので、CLI 自体の進捗テキストは制御しません（§4 と §7 の実例を参照）。さらに
  ターミナル上で `????` のように文字化けする場合は、それとは別の、純粋に表示上
  のエンコーディングの問題です（次の項目）。
- **Windows コンソールでキリル文字／日本語が文字化け** — 表示上のみ。ファイルと Web UI
  には影響しません。Web UI か UTF-8 端末を使うか、`sorta` 実行前に
  `PYTHONUTF8=1` を設定してください。
- **`sorta landmarks`（など）が「`data/landmarks.yaml` が見つからない」という
  相対パスのエラーで失敗する** — このパス（`config.yaml` の
  `naming.landmarks_file`）はリポジトリではなく**カレントディレクトリ**からの
  相対で解決されます。リポジトリのルートから `sorta` を実行するか、
  `config.yaml` の `naming.landmarks_file` に絶対パスを設定してください。

---

*Sorta はオリジナルを守り、ローカルで動作します。`--apply` の前に必ずプランを確認し、
問題があれば `sorta undo` を使ってください。*
