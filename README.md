# HAMSTER + ManiFlow 統合システム

**2Dパス表現と一貫性フロー学習を用いた階層型生成モデルによる汎用的なロボット操作**

*Hierarchical Action Models with 2D Paths and Consistency Flow Training for General Robot Manipulation*

> 本プロジェクトは、University of Washington の Hyak HPC 上で実施した研究のコードベースです。VLM (Vision-Language Model) ベースの高レベルパスプランナー **HAMSTER** と、Consistency Flow Matching ベースの低レベルアクションポリシー **ManiFlow** を統合し、ロボットマニピュレーションの汎化性能を検証しました。

---

## 目次

- [概要](#概要)
- [システムアーキテクチャ](#システムアーキテクチャ)
- [リポジトリ構成](#リポジトリ構成)
- [環境構築](#環境構築)
  - [必要な外部モデル・パッケージ](#必要な外部モデルパッケージ)
  - [Conda 環境セットアップ（メイン）](#conda-環境セットアップメイン)
  - [Docker / Singularity セットアップ（参考）](#docker--singularity-セットアップ参考)
- [データ準備](#データ準備)
  - [RoboTwin 2.0 でのデモデータ生成](#robotwin-20-でのデモデータ生成)
  - [HAMSTER パス生成](#hamster-パス生成)
  - [Zarr 形式への変換](#zarr-形式への変換)
- [学習](#学習)
- [評価](#評価)
- [実験結果](#実験結果)
  - [主要な知見](#主要な知見)
- [分析ツール](#分析ツール)
- [参考文献](#参考文献)

---

## 概要

ロボットマニピュレーションにおいて、大規模 VLM の豊かな世界知識と、小型ポリシーモデルの精密な運動制御を両立させることは重要な課題です。本研究では、以下の2つのモデルを階層的に統合するアプローチを提案・検証しました。

- **HAMSTER** (高レベル): VILA-1.5-13B を用いた 2D エンドエフェクタパス生成。RGB 画像とタスク記述を入力として、正規化された 2D ウェイポイント列とグリッパー状態を出力します。
- **ManiFlow** (低レベル): Consistency Flow Matching ベースの DiT-X (Diffusion Transformer with Cross-Attention) アクションポリシー。1-2 ステップの推論で高品質なアクション生成が可能です。

統合にあたり、HAMSTER が生成した 2D パスを ManiFlow の入力画像上にオーバーレイとして描画し、視覚的特徴として学習させます。さらに、エピソード開始時のパスを保持する **Memory Function** も提案しました。

RoboTwin 2.0 シミュレーション環境で 3 タスク × 6 条件の体系的な実験を行い、パスガイダンスの効果を In-domain / Cross-domain の両面から検証しています。

---

## システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                    推論フロー（16ステップごと）              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  RGB画像 + タスク記述（自然言語）                          │
│        │                                                │
│        ▼                                                │
│  ┌──────────────────────┐                               │
│  │  HAMSTER (VILA-1.5-13B)│  ← 高レベル: 2D パスプランナー │
│  │  FastAPI Server       │                               │
│  └──────────┬───────────┘                               │
│             │                                            │
│             ▼                                            │
│  2D パス: [(x1,y1), ..., (xN,yN)] + グリッパー状態        │
│             │                                            │
│             ▼                                            │
│  ┌──────────────────────┐                               │
│  │  パスオーバーレイ描画    │  jet colormap による時間的描画  │
│  └──────────┬───────────┘                               │
│             │                                            │
│             ▼                                            │
│  オーバーレイ画像 + ロボット状態 (14次元)                   │
│             │                                            │
│             ▼                                            │
│  ┌──────────────────────┐                               │
│  │  ManiFlow (DiT-X)     │  ← 低レベル: アクションポリシー  │
│  │  R3M + AdaLN-Zero     │                               │
│  │  12層, 8ヘッド, 768次元 │                               │
│  └──────────┬───────────┘                               │
│             │                                            │
│             ▼                                            │
│  ロボットアクション [horizon=16, 14次元]                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 学習の目的関数

ManiFlow は Consistency Flow Matching により、2つの損失関数を同時最適化します。

| 損失関数 | 重み | 役割 |
|---------|------|------|
| Flow Matching Loss | 75% | ノイズ→データの速度場を学習 |
| Consistency Loss | 25% | フロー軌道上の一貫性を保証（少ステップ推論を可能に） |

---

## リポジトリ構成

```
uw-project/
├── HAMSTER/                          # 高レベルパスプランナー
│   ├── server.py                     #   VILA-1.5-13B FastAPI サーバー (port 8000)
│   ├── server_qwen3.py               #   Qwen3-VL-8B FastAPI サーバー (port 8001)
│   ├── test_api_client.py            #   API テストクライアント + パスパーサー
│   ├── gradio_server_example.py      #   Gradio Web デモ
│   ├── setup_server.sh               #   VILA サーバー起動スクリプト
│   ├── setup_qwen3_server.sh         #   Qwen3 サーバー起動スクリプト
│   ├── start_vila_server_hyak.sh     #   Hyak 用 VILA 起動スクリプト
│   ├── scripts/
│   │   └── extract_frames_for_training.py  # HDF5 からフレーム抽出
│   └── tests/                        #   プロンプトエンジニアリングのテスト群 (36ファイル)
│
├── ManiFlow/                         # 低レベルアクションポリシー
│   ├── ManiFlow/maniflow/
│   │   ├── config/                   #   Hydra 設定ファイル群
│   │   │   ├── maniflow_original_robotwin2.yaml                  # 条件1,4用
│   │   │   ├── maniflow_overlay_current_robotwin2.yaml           # 条件2,5用
│   │   │   └── maniflow_overlay_image_policy_robotwin2_zarr.yaml # 条件3,6用
│   │   ├── dataset/
│   │   │   ├── robotwin2_original_zarr_dataset.py     # 素RGB画像データセット
│   │   │   ├── robotwin2_overlay_current_dataset.py   # Current overlay データセット
│   │   │   └── robotwin2_overlay_zarr_dataset.py      # Initial+Current overlay データセット
│   │   ├── model/
│   │   │   ├── diffusion/
│   │   │   │   ├── ditx.py                  # DiT-X Transformer バックボーン
│   │   │   │   ├── ditx_block.py            # DiT-X ブロック (adaLN-Zero)
│   │   │   │   └── ema_model.py             # EMA モデル
│   │   │   └── vision_2d/
│   │   │       └── multi_image_obs_encoder.py  # R3M/CLIP 2D 画像エンコーダ
│   │   ├── policy/
│   │   │   └── maniflow_image_policy.py       # 2D 画像ポリシー
│   │   ├── env/robotwin/               # RoboTwin 環境ラッパー
│   │   └── workspace/                  # 学習・評価ワークスペース
│   ├── scripts/
│   │   ├── generate_hamster_paths.py   # HAMSTER パス一括生成スクリプト
│   │   ├── convert_original_to_zarr.py # RGB画像 → Zarr 変換
│   │   └── convert_overlay_to_zarr.py  # オーバーレイ画像 → Zarr 変換
│   └── third_party/                    # 外部依存 (git管理外)
│       ├── RoboTwin2.0/               #   RoboTwin 2.0 シミュレータ
│       │   └── policy/ManiFlow_HAMSTER/  # 評価スクリプト群
│       ├── mujoco-py-2.1.2.14/        #   MuJoCo Python バインディング
│       ├── gym-0.21.0/                #   OpenAI Gym
│       ├── Metaworld/                 #   Meta-World ベンチマーク
│       ├── rrl-dependencies/          #   RRL 関連依存
│       └── r3m/                       #   R3M 視覚表現
│
├── analysis/                          # 実験結果の分析・可視化
│   ├── config.py                      #   6条件 × 3タスク × 4仮説の定義
│   ├── config_gt.py                   #   GT パス実験の設定
│   ├── generate_figures.py            #   論文用図表生成 (1092行)
│   ├── scripts/
│   │   ├── load_results.py            #   評価結果読み込みユーティリティ
│   │   └── utils.py                   #   共通ユーティリティ
│   ├── raw_data/                      #   評価結果の生データ (episodes.jsonl)
│   └── outputs/                       #   生成された図表・レポート
│
├── docker/                            # HPC 環境構築
│   ├── Dockerfile                     #   CUDA 12.1.1 + Ubuntu 22.04 ベース
│   ├── requirements_docker.txt        #   Python 依存関係 (93パッケージ)
│   ├── hyak_setup.sh                  #   Hyak Singularity セットアップ
│   ├── initialize_container.sh        #   コンテナ内初期化
│   └── README.md                      #   Docker/Hyak セットアップ手順
│
├── docs/                              # ドキュメント群
│   ├── TRAINING_GUIDE.md              #   学習手順の詳細ガイド
│   ├── EVALUATION_GUIDE.md            #   評価手順の詳細ガイド
│   ├── TRAINING_COMMANDS.md           #   学習コマンド集
│   ├── research/                      #   論文原稿・参考文献・図表
│   └── legacy/                        #   旧バージョンのドキュメント
│
├── IMPLEMENTATION_PLAN.md             # 実装計画書
├── PROJECT_PROGRESS.md                # 進捗管理 (57タスク完了, 進捗79%)
├── TASK_HISTORY.md                    # タスク履歴 (Phase 0-4)
└── .github/workflows/
    └── docker-build.yml               # Docker イメージ自動ビルド CI/CD
```

---

## 環境構築

### 必要な外部モデル・パッケージ

以下のモデル・パッケージはリポジトリに含まれておらず、別途取得が必要です。

#### VLM モデル（高レベルパスプランナー用）

| モデル | サイズ | 用途 | 取得方法 |
|--------|------|------|---------|
| **VILA-1.5-13B** (HAMSTER fine-tuned) | ~26GB VRAM | パス生成（メイン） | [HAMSTER プロジェクト](https://github.com/MLopezJ/HAMSTER)からモデルウェイトを取得 |
| Qwen3-VL-8B | ~17GB | パス生成（代替、性能低） | HuggingFace から自動ダウンロード (`Qwen/Qwen3-VL-8B`) |

> **注**: 論文の実験では VILA-1.5-13B を使用しています。Qwen3-VL-8B はゼロショット性能が低く、19バージョンのプロンプトエンジニアリングを経ても十分な性能に達しませんでした。

#### 外部パッケージ（`ManiFlow/third_party/` に配置、ソースからインストール）

| パッケージ | バージョン | 用途 | 取得方法 |
|-----------|----------|------|---------|
| **RoboTwin 2.0** | - | シミュレーション環境 | [RoboTwin GitHub](https://github.com/TianxingChen/RoboTwin) |
| **PyTorch3D** | 0.7.8 | 3D 処理 | ソースから editable インストール |
| **CuRobo** | - | GPU ベースロボット運動計画 | ソースから editable インストール |
| **r3m** | - | 視覚表現 (ResNet-18) | [R3M GitHub](https://github.com/facebookresearch/r3m)、ソースから editable インストール |

#### 主要な Python 依存関係

| パッケージ | バージョン | 用途 |
|-----------|----------|------|
| PyTorch | 2.6.0+cu124 | 深層学習フレームワーク |
| transformers | latest | VLM モデル読み込み |
| hydra-core | 1.3.2 | 設定管理 |
| sapien | 3.0.0b1 | SAPIEN 物理エンジン |
| zarr | 2.18.3 | データフォーマット |
| wandb | 0.23.1 | 実験ログ |
| openai | 2.14.0 | HAMSTER API クライアント |
| diffusers | 0.36.0 | 拡散モデルライブラリ |

完全な依存関係リストは [`docs/requirements_robotwin.txt`](docs/requirements_robotwin.txt) を参照してください。

### Hyak で GPU ノードを取得

```bash
# A40 GPU の場合（推奨）
srun -p gpu-a40 -A <lab_account> --nodes=1 --cpus-per-task=32 \
     --mem=400G --time=168:00:00 --gpus=2 --pty /bin/bash

# L40s GPU の場合
srun -p gpu-l40s -A <lab_account> --nodes=1 --cpus-per-task=120 \
     --mem=1000G --time=24:00:00 --gpus=6 --pty /bin/bash
```

> **注**: 評価時は A40 GPU が必要です。L40s / H200 では cuRobo の互換性問題が発生します。

### Conda 環境セットアップ（メイン）

Hyak 上では Conda の `robotwin` 環境を構築して実行します。

#### 1. Miniconda のインストール

```bash
cd /gscratch/scrubbed/${USER}

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /gscratch/scrubbed/${USER}/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

source /gscratch/scrubbed/${USER}/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc
```

#### 2. 環境の作成と基本パッケージのインストール

```bash
# Python 3.10 環境を作成
conda create -n robotwin python=3.10.19 -y
conda activate robotwin

# ffmpeg のインストール（動画処理用）
conda install -c conda-forge ffmpeg=7.1 x264 -y

# PyTorch のインストール（CUDA 12.4）
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# その他の依存関係
pip install -r docs/requirements_robotwin.txt
```

#### 3. ソースパッケージのインストール（editable）

```bash
# CUDA モジュールのロード（Hyak 固有）
module load cuda/12.4.1 gcc/13.2.0

PROJECT_DIR="/gscratch/scrubbed/${USER}/projects/HAMSTER-ManiFlow-Integration"

# PyTorch3D
cd ${PROJECT_DIR}/ManiFlow/third_party/pytorch3d
pip install -e .

# CuRobo
cd ${PROJECT_DIR}/ManiFlow/third_party/curobo
pip install -e src/

# R3M（視覚表現エンコーダ）
cd ${PROJECT_DIR}/ManiFlow/third_party/r3m
pip install -e .

# ManiFlow 本体
cd ${PROJECT_DIR}/ManiFlow/ManiFlow
pip install -e .
```

#### 4. パッチの適用

```bash
# Patch 1: PyTorch cpp_extension.py の CUDA バージョンチェックを warning に変更
TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
CPP_EXT="${TORCH_PATH}/utils/cpp_extension.py"
cp "${CPP_EXT}" "${CPP_EXT}.backup"
# RuntimeError → warnings.warn に変更（_check_cuda_version() 内）

# Patch 2: CuRobo の設定ファイルでアセットパスを絶対パスに置換
CUROBO_PATH="${PROJECT_DIR}/ManiFlow/third_party/curobo"
find "${CUROBO_PATH}/src/curobo/content" -name "*.yml" -exec sed -i \
    "s|\${ASSETS_PATH}|${CUROBO_PATH}/src/curobo/content/assets|g" {} \;
```

#### 5. インストールの検証

```bash
source /gscratch/scrubbed/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
module load cuda/12.4.1 gcc/13.2.0

python -c "import torch; print(f'torch=={torch.__version__}')"
python -c "import sapien; print(f'sapien=={sapien.__version__}')"
python -c "import pytorch3d; print(f'pytorch3d=={pytorch3d.__version__}')"
python -c "import maniflow; print('maniflow OK')"
python -c "import r3m; print('r3m OK')"
```

詳細は [`docs/CONDA_ENVIRONMENT_SETUP.md`](docs/CONDA_ENVIRONMENT_SETUP.md) を参照してください。

### Docker / Singularity セットアップ（参考）

プロジェクト初期には Docker イメージを Singularity コンテナとして Hyak 上で利用していました。Conda 環境が推奨ですが、Docker ベースのセットアップも `docker/` ディレクトリに残してあります。

```bash
# ローカルでビルド & プッシュ
cd docker/
docker build -t <your-dockerhub-username>/hamster-maniflow:latest .
docker push <your-dockerhub-username>/hamster-maniflow:latest

# Hyak 上で Singularity として取得・起動
module load singularity
singularity pull docker://<your-dockerhub-username>/hamster-maniflow:latest
singularity instance start --nv --bind /gscratch/:/gscratch/:rw \
    hamster-maniflow_latest.sif hamster_train
```

詳細は [`docker/README.md`](docker/README.md) を参照してください。

---

## データ準備

### RoboTwin 2.0 でのデモデータ生成

RoboTwin 2.0 のスクリプトポリシーを使って、各タスクの学習データ（エキスパートデモ）を生成します。

```bash
# RoboTwin 2.0 のセットアップ後
cd ManiFlow/third_party/RoboTwin2.0

# 各タスク × 環境で 50 エピソードずつ生成
# clean 環境
python collect_data.py --task beat_block_hammer --env clean --episodes 50
python collect_data.py --task click_bell --env clean --episodes 50
python collect_data.py --task move_can_pot --env clean --episodes 50

# cluttered 環境
python collect_data.py --task beat_block_hammer --env cluttered --episodes 50
python collect_data.py --task click_bell --env cluttered --episodes 50
python collect_data.py --task move_can_pot --env cluttered --episodes 50
```

生成されるデータ構造:
```
HAMSTER/results/evaluation_tasks_{clean,cluttered}/{task}/episode_XX/
├── frames/            # 素の RGB 画像 (head_camera)
├── paths/             # パス座標 (.pkl)
└── overlay_images/    # オーバーレイ画像
```

### HAMSTER パス生成

VILA サーバーを起動し、各エピソードの初期フレームに対してパスを生成します。

#### 1. VILA サーバーの起動

```bash
cd HAMSTER
bash setup_server.sh  # port 8000 で VILA-1.5-13B が起動
```

#### 2. パスの一括生成

```bash
cd ManiFlow

python scripts/generate_hamster_paths.py \
    --zarr-path /path/to/dataset.zarr \
    --output-path /path/to/hamster_paths.pkl \
    --task-description "there is a hammer and a block on the table, use the arm to grab the hammer and beat the block" \
    --server-ip 127.0.0.1 \
    --server-port 8000 \
    --resume  # 中断時の再開に対応
```

#### パス出力フォーマット

VILA は以下の形式で 2D パスを出力します:
```
<ans>[(x1, y1), (x2, y2), <action>Close Gripper</action>, (x3, y3), ...]</ans>
```
- 座標は [0, 1] に正規化された相対位置
- `<action>` タグでグリッパー状態の変化点を示す
- パース後は各ウェイポイントが `[x, y, gripper_state]` の形式に変換される

#### パス生成の信頼性対策

- **リトライ機構**: フレームごとに最大2回までリトライ
- **フォールバックパース**: `<ans>` タグ省略時の自動補完
- **フォールバックパス**: 生成失敗時は直前の成功パスを代用

### Zarr 形式への変換

学習データを ManiFlow が読み込める Zarr 形式に変換します。

```bash
cd ManiFlow/ManiFlow

# オリジナル画像（条件1,4用）
python scripts/convert_original_to_zarr.py \
    --input-dir /path/to/evaluation_tasks_clean \
    --output data/zarr/clean_original_beat_block_hammer.zarr \
    --tasks beat_block_hammer --episodes 50

# オーバーレイ画像（条件2,3,5,6用）
python scripts/convert_overlay_to_zarr.py \
    --overlay-base /path/to/evaluation_tasks_clean \
    --output data/zarr/clean_overlay_beat_block_hammer.zarr \
    --tasks beat_block_hammer --episodes 50
```

**Zarr ファイルの命名規則**: `{env}_{type}_{task}.zarr`
- `env`: `clean` または `cluttered`
- `type`: `original` または `overlay`
- `task`: タスク名

---

## 学習

### 実験条件マトリクス

|  | Original ManiFlow | Overlay (current) | Overlay (initial+current) |
|--|-------------------|--------------------|---------------------------|
| **学習: cluttered** | 条件1 (C1) | 条件2 (C2) | 条件3 (C3) |
| **学習: clean** | 条件4 (C4) | 条件5 (C5) | 条件6 (C6) |

全条件で **cluttered table** 環境で評価。

### 共通の学習パラメータ

| パラメータ | 値 |
|-----------|-----|
| Horizon | 16 ステップ |
| Observation steps | 2 |
| Action steps | 16 |
| Batch size | 64 |
| Epochs | 501 |
| Optimizer | AdamW (lr=1e-4, betas=[0.9, 0.95]) |
| LR Scheduler | Cosine (warmup 500 steps) |
| Vision Encoder | R3M (ResNet-18) |
| EMA decay | 0.999 |
| Flow/Consistency ratio | 75% / 25% |

### 学習の実行

```bash
cd ManiFlow/ManiFlow

# 条件1: cluttered + original（ベースライン）
./scripts/train_original.sh cluttered beat_block_hammer 0 42

# 条件2: cluttered + overlay current
./scripts/train_overlay_current.sh cluttered beat_block_hammer 0 42

# 条件3: cluttered + overlay initial+current（Memory Function）
./scripts/train_overlay_initial_current.sh cluttered beat_block_hammer 0 42

# 条件4-6: clean 環境（第1引数を clean に変更）
./scripts/train_original.sh clean beat_block_hammer 0 42
./scripts/train_overlay_current.sh clean beat_block_hammer 0 42
./scripts/train_overlay_initial_current.sh clean beat_block_hammer 0 42
```

引数: `<env> <task> <gpu_id> <seed>`

チェックポイントは `ManiFlow/ManiFlow/data/outputs/` 以下に保存されます。評価には Epoch 500 のチェックポイントを使用します。

詳細は [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) を参照してください。

---

## 評価

### 評価の実行

```bash
cd ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER

# 条件2,3,5,6 では事前に VILA サーバーを起動
bash start_vila_server.sh 0 8000  # GPU 0, port 8000

# 評価実行（100エピソード）
bash eval.sh --task click_bell --mode original --env cluttered --seed 42 --episodes 100       # C1
bash eval.sh --task click_bell --mode current --env cluttered --seed 42 --episodes 100        # C2
bash eval.sh --task click_bell --mode initial_current --env cluttered --seed 42 --episodes 100 # C3
bash eval.sh --task click_bell --mode original --env clean --seed 42 --episodes 100           # C4
bash eval.sh --task click_bell --mode current --env clean --seed 42 --episodes 100            # C5
bash eval.sh --task click_bell --mode initial_current --env clean --seed 42 --episodes 100    # C6

# VILA サーバーの停止
bash stop_vila_server.sh
```

### 評価フロー

- **Original モード**: RGB画像 → ManiFlow → 16ステップのアクション
- **Current / Initial+Current モード**: RGB画像 → VILA → パス生成 → オーバーレイ描画 → ManiFlow → 16アクション（16ステップごとにパス再生成）

### 評価結果の出力

```
eval_results/{task}/condition{N}_{train_env}_{mode}_eval{eval_env}/
└── run_seed{seed}_{run_number}/
    ├── episodes.jsonl    # 各エピソードの詳細ログ（成功/失敗、パス統計、推論時間）
    └── eval_results/     # 動画ファイル
```

詳細は [`docs/EVALUATION_GUIDE.md`](docs/EVALUATION_GUIDE.md) を参照してください。

---

## 実験結果

### タスク成功率 (%)

| タスク | C1 | C2 | C3 | C4 | C5 | C6 |
|--------|-----|-----|-----|-----|------|-----|
| Click Bell | 4.0 | 2.0 | 4.0 | 3.0 | **20.0** | 1.0 |
| Move Can Pot | 21.0 | 20.0 | 26.0 | 27.0 | 27.0 | 27.0 |
| Beat Block Hammer | 4.0 | 6.0 | 1.0 | 31.0 | **37.0** | 11.0 |
| **平均** | 9.7 | 9.3 | 10.3 | 20.3 | **28.0** | 13.0 |

- C1-C3: Cluttered 環境で学習 → Cluttered 環境で評価（In-domain）
- C4-C6: Clean 環境で学習 → Cluttered 環境で評価（Cross-domain）

### 推論時間

| 条件 | VILA (ms) | ManiFlow (ms) | オーバーヘッド |
|------|----------|--------------|-------------|
| C1, C4 (パスなし) | - | 108-136 | - |
| C2, C5 (current) | 4,313-4,703 | 118-136 | +4.3-4.7s / 16steps |
| C3, C6 (initial+current) | 4,361-4,363 | 118-126 | +4.4s / 16steps |

### 主要な知見

#### 1. Clean 環境学習の優位性（予想外の発見）

Clean 環境で学習したモデル (C4-C6, 平均 13.0-28.0%) が、Cluttered 環境で学習したモデル (C1-C3, 平均 9.3-10.3%) を一貫して上回りました。視覚的にシンプルな環境での学習が、タスクに本質的な運動パターンの獲得を促進し、未見の Cluttered 環境への汎化性能を向上させたと考えられます。

#### 2. VLM パスガイダンスの Cross-domain 効果 (H2: 支持)

C5 (Current path, Cross-domain) は C4 (ベースライン) に対して**平均 +7.7%** の改善を達成。VILA のパスが、学習環境 (clean) と評価環境 (cluttered) の視覚的ギャップを補償する意味的ガイダンスとして機能しました。

#### 3. Memory Function の予想外の性能低下 (H3: 不支持)

Initial + Current パス（Memory Function）は、特に Cross-domain 条件 (C5→C6) で**-15.0%** の大幅な性能低下を引き起こしました。2つのオーバーレイ画像の入力が情報過多となり、Clean 環境で学習したモデルにとって負担になったと考えられます。

#### 4. In-domain でのパスガイダンスの限定的効果 (H1: 不支持)

In-domain 条件 (C1→C2,C3) では、パスガイダンスによる一貫した改善は見られませんでした。同一環境での学習・評価では、パスオーバーレイが視覚的ノイズとして作用した可能性があります。

---

## 分析ツール

`analysis/` ディレクトリには、論文用の図表を生成するためのスクリプト群が含まれています。

```bash
cd analysis

# 論文用図表の一括生成
python generate_figures.py
```

主要なスクリプト:
- `generate_figures.py`: メイン結果のグラフ、仮説検証図、パス統計、サマリーフィギュアなど
- `config.py`: 6条件 × 3タスク × 4仮説の実験設定定義
- `scripts/load_results.py`: `episodes.jsonl` からのデータ読み込みユーティリティ

出力先: `analysis/outputs/figures/`, `analysis/outputs/tables/`

---

## 参考文献

- **HAMSTER**: Li et al., "HAMSTER: Hierarchical Action Models for Open-World Robot Manipulation" ([paper](https://arxiv.org/abs/2501.12604))
- **ManiFlow**: Yan et al., "ManiFlow: Implicitly Representing Manipulation Spaces for Robot Manipulation via Consistency Flow Matching" ([paper](https://arxiv.org/abs/2503.09950))
- **VILA**: Lin et al., "VILA: On Pre-training for Visual Language Models" ([paper](https://arxiv.org/abs/2312.07533))
- **RoboTwin 2.0**: Chen et al., "RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins" ([paper](https://arxiv.org/abs/2504.13059))
- **Consistency Models**: Song et al., "Consistency Models" ([paper](https://arxiv.org/abs/2303.01469))
- **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling" ([paper](https://arxiv.org/abs/2210.02747))
- **R3M**: Nair et al., "R3M: A Universal Visual Representation for Robot Manipulation" ([paper](https://arxiv.org/abs/2203.12601))

