# ManiFlow トレーニングガイド

**最終更新**: 2025-12-21

## 概要

本ガイドでは、3種類のManiFlowモデルをRoboTwin 2.0データでトレーニングする手順を説明する。

### モデル構成

| モデル名 | 入力画像 | 実験条件 | 用途 |
|---------|---------|---------|------|
| **Original** | 素のRGB画像 | 条件1,4 | ベースライン |
| **Overlay (current)** | 現在フレームのオーバーレイ画像 | 条件2,5 | パスガイダンス効果検証 |
| **Overlay (initial+current)** | 初期+現在フレームのオーバーレイ画像 | 条件3,6 | Memory Function効果検証 |

---

## クイックスタート

### 1. 前提条件

- パス生成済みデータが存在すること
- 以下のディレクトリ構造があること:

```
HAMSTER/results/evaluation_tasks_{clean,cluttered}/{task}/episode_XX/
├── frames/           # 素のRGB画像
├── paths/            # パス座標 (.pkl)
└── overlay_images/   # オーバーレイ画像（generate_overlay_images.pyで生成）
```

### 2. データ準備（Zarr変換）

**命名規則**: `{env}_{type}_{task}.zarr`
- `env`: `clean` または `cluttered`
- `type`: `original` または `overlay`
- `task`: タスク名（例: `beat_block_hammer`, `click_bell`）

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow

# オーバーレイ版Zarr（条件2,3,5,6用）- タスクごとに作成
python scripts/convert_overlay_to_zarr.py \
    --overlay-base /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/evaluation_tasks_clean \
    --output data/zarr/clean_overlay_beat_block_hammer.zarr \
    --tasks beat_block_hammer --episodes 50

# オリジナル版Zarr（条件1,4用）- タスクごとに作成
python scripts/convert_original_to_zarr.py \
    --input-dir /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/evaluation_tasks_clean \
    --output data/zarr/clean_original_beat_block_hammer.zarr \
    --tasks beat_block_hammer --episodes 50
```

### 3. トレーニング実行

GPUノード上で直接実行する。各GPUノードで以下のコマンドを実行。

**環境準備とバックグラウンド実行の完全なコマンド:**

```bash
# 条件1: cluttered + original（ベースライン）
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh cluttered beat_block_hammer 0 42 &

# 条件2: cluttered + overlay current
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh cluttered beat_block_hammer 0 42 &

# 条件3: cluttered + overlay initial+current（Memory Function）
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh cluttered beat_block_hammer 0 42 &

# 条件4: clean + original（ベースライン）
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh clean beat_block_hammer 0 42 &

# 条件5: clean + overlay current
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh clean beat_block_hammer 0 42 &

# 条件6: clean + overlay initial+current（Memory Function）
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh clean beat_block_hammer 0 42 &
```

**コマンドの構成要素:**
- `source ... && conda activate robotwin`: Conda環境の有効化
- `module load cuda/12.4.1 gcc/13.2.0`: CUDAとGCCモジュールのロード（必須）
- `WANDB_MODE=disabled`: W&Bロギングの無効化
- `nohup ... &`: バックグラウンド実行（SSHセッション切断後も継続）

**ログの確認:**
```bash
tail -f /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/logs/train_*.log
```

**プロセスの確認:**
```bash
ps aux | grep train
nvidia-smi  # GPU使用状況
```

---

## ディレクトリ構成

### データ

```
ManiFlow/ManiFlow/data/
├── zarr/
│   ├── {env}_{type}_{task}.zarr   # タスクごとに個別Zarr
│   │   例:
│   │   ├── clean_original_beat_block_hammer.zarr
│   │   ├── clean_overlay_beat_block_hammer.zarr
│   │   ├── cluttered_original_beat_block_hammer.zarr
│   │   ├── cluttered_overlay_beat_block_hammer.zarr
│   │   ├── clean_original_click_bell.zarr
│   │   └── ...
└── outputs/                        # トレーニング済みチェックポイント
```

### 設定ファイル

```
ManiFlow/ManiFlow/maniflow/config/
├── # ポリシー設定
├── maniflow_original_robotwin2.yaml              # 条件1,4: オリジナル
├── maniflow_overlay_current_robotwin2.yaml       # 条件2,5: current only
├── maniflow_overlay_image_policy_robotwin2_zarr.yaml  # 条件3,6: initial+current
│
└── robotwin_task/
    ├── robotwin2_original.yaml                   # オリジナル用タスク設定
    ├── robotwin2_overlay_current.yaml            # current only用タスク設定
    └── robotwin2_overlay_single_arm_zarr.yaml    # initial+current用タスク設定
```

### データセット

```
ManiFlow/ManiFlow/maniflow/dataset/
├── robotwin2_original_zarr_dataset.py      # 条件1,4用: オリジナルRGB
├── robotwin2_overlay_current_dataset.py    # 条件2,5用: current only
└── robotwin2_overlay_zarr_dataset.py       # 条件3,6用: initial+current
```

### スクリプト

```
ManiFlow/ManiFlow/scripts/
├── # Zarr変換
├── convert_original_to_zarr.py             # オリジナル画像のZarr変換
├── convert_overlay_to_zarr.py              # オーバーレイ画像のZarr変換
│
├── # トレーニング（GPUノード上で直接実行）
├── train_original.sh                       # 条件1,4用
├── train_overlay_current.sh                # 条件2,5用
└── train_overlay_initial_current.sh        # 条件3,6用
```

---

## Zarr変換

### オーバーレイ版（条件2,3,5,6用）

`overlay_images/` からオーバーレイ画像を読み込み、Zarr形式に変換する。

```bash
/usr/bin/singularity exec --bind /gscratch/:/gscratch/:rw --bind /mmfs1/:/mmfs1/:rw \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    python scripts/convert_overlay_to_zarr.py \
    --overlay-base /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/evaluation_tasks_clean \
    --output data/zarr/clean_overlay_beat_block_hammer.zarr \
    --tasks beat_block_hammer --episodes 50
```

**Zarr構造**:
```
clean_overlay_beat_block_hammer.zarr/
├── data/
│   ├── overlay_image    # (N, 3, 224, 224) float32
│   ├── action           # (N, 14) float32
│   └── state            # (N, 14) float32
└── meta/
    └── episode_ends     # (num_episodes,) int64
```

### オリジナル版（条件1,4用）

`frames/` から素のRGB画像を読み込み、Zarr形式に変換する。

```bash
/usr/bin/singularity exec --bind /gscratch/:/gscratch/:rw --bind /mmfs1/:/mmfs1/:rw \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    python scripts/convert_original_to_zarr.py \
    --input-dir /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/evaluation_tasks_clean \
    --output data/zarr/clean_original_beat_block_hammer.zarr \
    --tasks beat_block_hammer --episodes 50
```

**Zarr構造**:
```
clean_original_beat_block_hammer.zarr/
├── data/
│   ├── image            # (N, 3, 224, 224) float32
│   ├── action           # (N, 14) float32
│   └── state            # (N, 14) float32
└── meta/
    └── episode_ends     # (num_episodes,) int64
```

---

## トレーニング

### 共通設定

| パラメータ | 値 |
|-----------|-----|
| horizon | 16 |
| n_obs_steps | 2 |
| n_action_steps | 16 |
| batch_size | 64 |
| num_epochs | 501 |
| optimizer | AdamW (lr=1e-4) |
| encoder | R3M (ResNet-18) |

### 条件1,4: オリジナルManiFlow

素のRGB画像を入力として使用。

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow

# GPU 0, seed 42で実行
./scripts/train_original.sh 0 42
```

**入力**:
- `image`: 現在フレームのRGB画像 [B, T, 3, 224, 224]
- `agent_pos`: ロボット状態 [B, T, 14]

### 条件2,5: Overlay (current only)

現在フレームのオーバーレイ画像のみを入力として使用。

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow

# GPU 0, seed 42で実行
./scripts/train_overlay_current.sh 0 42
```

**入力**:
- `current_overlay`: 現在フレームのオーバーレイ画像 [B, T, 3, 224, 224]
- `agent_pos`: ロボット状態 [B, T, 14]

### 条件3,6: Overlay (initial + current)

初期フレームと現在フレームのオーバーレイ画像を入力として使用（Memory Function）。

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow

# GPU 0, seed 42で実行
./scripts/train_overlay_initial_current.sh 0 42
```

**入力**:
- `initial_overlay`: 初期フレーム(frame 0)のオーバーレイ画像 [B, T, 3, 224, 224]
- `current_overlay`: 現在フレームのオーバーレイ画像 [B, T, 3, 224, 224]
- `agent_pos`: ロボット状態 [B, T, 14]

---

## チェックポイント

### 出力先

```
ManiFlow/ManiFlow/data/outputs/{date}/{time}_{config_name}/
├── checkpoints/
│   ├── latest.ckpt
│   └── epoch=XXXX-val_loss=X.XXXXXX.ckpt
├── config.yaml
└── logs/
```

### チェックポイント管理

| ファイル | 説明 |
|---------|------|
| `latest.ckpt` | 最新のチェックポイント |
| `epoch=XXXX-val_loss=X.XXXXXX.ckpt` | Top-K保存（val_lossが低いもの） |

---

## 実験条件マトリクス

### 条件一覧

|  | オリジナルManiFlow | Overlay (current) | Overlay (initial+current) |
|--|-------------------|-------------------|---------------------------|
| **学習: cluttered** | 条件1 | 条件2 | 条件3 |
| **学習: clean** | 条件4 | 条件5 | 条件6 |

**評価**: 全条件で cluttered table で評価

### 必要なZarrファイル（タスクごと）

| 条件 | Zarrファイル（例: beat_block_hammer） |
|------|--------------------------------------|
| 条件1 | `cluttered_original_beat_block_hammer.zarr` |
| 条件2,3 | `cluttered_overlay_beat_block_hammer.zarr` |
| 条件4 | `clean_original_beat_block_hammer.zarr` |
| 条件5,6 | `clean_overlay_beat_block_hammer.zarr` |

---

## 実装状況

### Zarr変換スクリプト

- [x] `convert_overlay_to_zarr.py` - オーバーレイ画像のZarr変換
- [x] `convert_original_to_zarr.py` - オリジナル画像のZarr変換

### データセット

- [x] `robotwin2_overlay_zarr_dataset.py` - initial+current用データセット（条件3,6）
- [x] `robotwin2_overlay_current_dataset.py` - current only用データセット（条件2,5）
- [x] `robotwin2_original_zarr_dataset.py` - オリジナル用データセット（条件1,4）

### 設定ファイル

- [x] `maniflow_overlay_image_policy_robotwin2_zarr.yaml` - initial+current用設定
- [x] `maniflow_overlay_current_robotwin2.yaml` - current only用設定
- [x] `maniflow_original_robotwin2.yaml` - オリジナル用設定
- [x] `robotwin_task/robotwin2_overlay_single_arm_zarr.yaml` - initial+current用タスク設定
- [x] `robotwin_task/robotwin2_overlay_current.yaml` - current only用タスク設定
- [x] `robotwin_task/robotwin2_original.yaml` - オリジナル用タスク設定

### トレーニングスクリプト

- [x] `train_overlay_initial_current.sh` - initial+current用（条件3,6）
- [x] `train_overlay_current.sh` - current only用（条件2,5）
- [x] `train_original.sh` - オリジナル用（条件1,4）

### Zarrファイル生成状況

| タスク | clean_original | clean_overlay | cluttered_original | cluttered_overlay |
|--------|----------------|---------------|--------------------|--------------------|
| beat_block_hammer | ✅ | ✅ | ✅ | ✅ |
| click_bell | ✅ | ✅ | ✅ | ✅ |

### TODO

- [x] Zarrファイルの生成（beat_block_hammer, click_bell完了）
- [ ] 各条件でのトレーニング実行
- [ ] 評価パイプラインとの統合

---

## トラブルシューティング

### num_workers問題

Zarrデータセットでは `num_workers=0` を使用すること。ランダムアクセスパターンでデッドロックが発生する可能性がある。

```yaml
dataloader:
  num_workers: 0
  persistent_workers: false
```

### R3Mモジュールが見つからない

```bash
--env PYTHONPATH="/mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/r3m"
```

### tqdm出力がバッファリングされる

```bash
--env PYTHONUNBUFFERED=1
```

---

## 関連ドキュメント

- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - 評価手順
- [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md) - 研究概要
- [NEXT_PHASE_REQUIREMENTS.md](NEXT_PHASE_REQUIREMENTS.md) - 詳細な実装計画
