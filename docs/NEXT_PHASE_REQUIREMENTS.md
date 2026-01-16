# Next Phase Requirements: ManiFlow + Path Input Training

## 現状の進捗

- Qwen3-VLのパス生成能力を評価完了
- シングルアーム + オブジェクト可視状態では高精度
- RoboTwin 1.0動画での時間的ロバスト性を確認

---

## 次のステップ：ManiFlow + パス入力での学習

### 基本アプローチ（シンプル版）

- タスク開始時（最初のフレーム）のみでパス生成
- 以降はそのパスをManiFlowへの入力として固定使用
- **問題点**: タスク実行中の環境変化（オクルージョン等）に対応不可

### 新アプローチ：画像ベースのパス統合（2025-12-08 更新）

**設計思想**:
- ManiFlowは本来**画像ベースのImitation Learningモデル**である
- パスを画像に統合する手法は、VLA研究で標準化しつつあるアプローチ
- 他のVLAモデル（OpenVLA, RT-2等）への転用可能性を考慮

2つのアプローチ：

#### A. オーバーレイ画像入力

```
入力: [Initial Overlay Image] + [Current Overlay Image] + Proprio + Language → ManiFlow
```

**オーバーレイ画像の定義**:
- 元のRGB画像上に2Dパスを直接描画
- パスは時間経過を表す**グラデーション線**で表現
- グリッパーの開閉状態を**円のマーカー**で表現

**特徴**:
- 視覚情報とパス情報が1つの画像に統合
- 他のVLAモデルにそのまま転用可能
- 人間が視覚的にデバッグしやすい
- 既存の画像エンコーダをそのまま使用可能

#### B. Concat画像入力（6チャンネル）

```
入力: [Initial Concat Image] + [Current Concat Image] + Proprio + Language → ManiFlow
```

**Concat画像の定義**:
- RGB画像（3チャンネル）+ パスのみ画像（3チャンネル）= 6チャンネル
- パスのみ画像: 黒背景にパスだけを描画

**特徴**:
- RGBとパス情報がチャンネルレベルで分離
- モデルが「視覚」と「パス」を区別しやすい
- 入力層の変更が必要（3ch → 6ch）

#### A vs B 比較

| | A: オーバーレイ | B: Concat 6ch |
|--|----------------|---------------|
| 入力形式 | RGB画像 [3, H, W] × 2 | 6ch画像 [6, H, W] × 2 |
| 情報分離 | 混在（同一画像上） | チャンネルで分離 |
| 他モデル転用 | ✅ そのまま使える | ⚠️ 入力層変更必要 |
| 実装難易度 | 低い | やや高い |
| デバッグ容易性 | ✅ 視覚的に確認可能 | ✅ まあまあ |

**共通の利点**:
- 画像ベースなのでManiFlowの本来の設計に沿っている
- Initial画像でメモリ機構を実現（オクルージョン対策）
- Language条件付けでマルチタスク対応
- Imitation Learningなので事前学習は不要（スクラッチから学習）

**注**: どちらの場合も、オクルージョンの明示的な検出は不要。モデルが学習データから「見えてる時は現在の観測を重視、見えない時は過去の情報を重視」することを自動的に学ぶ想定。

### 本質的な課題

オクルージョン発生時に「見えなくなった物体がどこにあるか」を推論する必要がある。メモリ機構でこれを解決する。

---

## 実験方針

### Phase 1: シンプルなタスクで検証

- まずはシンプルなタスク（シングルアーム、往復なし）のみを使用
- 基本アプローチとメモリ機構の効果を比較検証

### Phase 2: 複雑なタスクへの拡張（将来）

- 複雑なタスク（アームが複数回往復する等）は、簡単なサブタスクに分割
- 各サブタスクに対してこの手法を適用
- 例：「物体Aを拾って置く → 物体Bを拾って置く」を2つのサブタスクとして処理

---

## TODO

- [x] 具体的な実装方針の決定 → ~~旧方針: Dual Path座標入力~~ → **新方針: 画像ベースアプローチに変更（2025-12-08）**
- [x] 使用するタスクの選定 → **RoboTwin 2.0 シングルアームタスク**
- [x] VILAによるパス生成完了（episode_00のみ、2025-12-08）
- [x] 全50エピソードに対するVILAパス生成 → **4GPU並列実行で進行中（2025-12-10）** → **~36%完了（19,481/53,553フレーム）**
- [x] オーバーレイ画像生成スクリプトの実装 → **`HAMSTER/tests/training_data/` に実装完了（2025-12-10）**
- [x] **アプローチA（オーバーレイ）** のデータセット・ポリシー実装 → **実装完了（2025-12-10）**
- [x] 統合テスト（Dataset, DataLoader, Policy初期化）→ **CPUテスト完了（2025-12-10）**
- [x] **データローディング問題の解決** → **Zarr変換方式に移行（2025-12-11）**
  - PNG版データセットは1エポック1時間以上かかる問題を発見
  - 原因: 毎回のディスクI/O（cv2.imread + HDF5 open）
  - 解決: Zarr形式に事前変換してメモリにロード
- [x] Zarr変換の実行 → **完了（2025-12-12）**
  - 150エピソード、17,255フレーム、約10.4GB
- [x] GPUでのForward/Backward passテスト → **完了（2025-12-12）**
- [ ] アプローチAでの訓練・検証 ← **訓練正常進行中（2025-12-13）**
  - 501エポック、1エポック約3分（num_workers=0で高速化）
  - 推定完了時間: 約25時間
  - ~~**⚠️ デッドロック問題**: Epoch 2 batch 178/267で停止（GPU使用率0%）~~
  - **解決済み**: デッドロックではなくtqdm出力のバッファリング問題だった（詳細は下記参照）
- [ ] （Aが不十分な場合のみ）アプローチB（Concat 6ch）の実装・検証
- [ ] 評価指標の定義
- [ ] 実験計画の策定

---

## 実装計画: アプローチA（オーバーレイ画像入力）

**決定日**: 2025-12-08（旧方針から変更）

### 採用方針

- **アプローチA（オーバーレイ画像入力）** を採用
- **全フレームでパス生成 → オーバーレイ画像生成**
- **RoboTwin 2.0** のシングルアームタスクを使用
- **ManiFlowの画像ベースポリシー**（`ManiFlowTransformerImagePolicy`）を使用

### データフロー

```
[RoboTwin 2.0 HDF5]
        ↓
[VILA-1.5-13B (HAMSTER finetuned)] → 全フレームのパスを事前生成
        ↓
[Overlay Image Generator]
  - RGB画像 + パス座標 → オーバーレイ画像
  - パス描画: グラデーション線（時間経過）+ 円マーカー（グリッパー状態）
        ↓
[RoboTwin2OverlayDataset]
  - initial_overlay: 初期フレーム(frame 0)のオーバーレイ画像 [B, T, 3, H, W]
  - current_overlay: 現在フレームのオーバーレイ画像 [B, T, 3, H, W]
  - agent_pos: ロボット状態 [B, T, 14]
  - task_name: タスク名（Language条件付け用、将来対応）
        ↓
[TimmObsEncoder (R3M / ResNet-18)]
  - R3Mはロボット操作動画で事前学習済み（ManiFlowオリジナル設定）
  - initial_overlay → 512次元 visual features
  - current_overlay → 512次元 visual features
  - agent_pos → 14次元
  - 結合して Visual Tokens へ (合計: 512 * n_obs_steps * 2 + 14 * n_obs_steps)
        ↓
[DiTX (8層, 8ヘッド, 512次元)]
  - flow_batch_ratio: 0.75
  - consistency_batch_ratio: 0.25
        ↓
[Consistency Flow (10ステップODE)] → Actions [B, horizon, 14]
```

### 新規作成ファイル

| ファイル | 場所 | 目的 | 状態 |
|---------|------|------|------|
| `generate_overlay_images.py` | `HAMSTER/tests/training_data/` | パス座標からオーバーレイ画像を生成 | ✅ 実装完了 |
| `overlay_utils.py` | `HAMSTER/tests/training_data/` | HAMSTER準拠のオーバーレイ描画ユーティリティ | ✅ 実装完了 |
| `robotwin2_overlay_dataset.py` | `maniflow/dataset/` | RoboTwin 2.0 オーバーレイ画像データセット（PNG版、非推奨） | ✅ 実装完了（2025-12-10） |
| `robotwin2_overlay_zarr_dataset.py` | `maniflow/dataset/` | RoboTwin 2.0 オーバーレイ画像データセット（**Zarr版、推奨**） | ✅ 実装完了（2025-12-11） |
| `convert_overlay_to_zarr.py` | `ManiFlow/scripts/` | PNG+HDF5 → Zarr変換スクリプト | ✅ 実装完了（2025-12-11） |
| `robotwin2_overlay_single_arm.yaml` | `maniflow/config/robotwin_task/` | タスク設定（PNG版） | ✅ 実装完了（2025-12-10） |
| `robotwin2_overlay_single_arm_zarr.yaml` | `maniflow/config/robotwin_task/` | タスク設定（**Zarr版、推奨**） | ✅ 実装完了（2025-12-11） |
| `maniflow_overlay_image_policy_robotwin2.yaml` | `maniflow/config/` | 訓練用ポリシー設定（PNG版） | ✅ 実装完了（2025-12-10） |
| `maniflow_overlay_image_policy_robotwin2_zarr.yaml` | `maniflow/config/` | 訓練用ポリシー設定（**Zarr版、推奨**） | ✅ 実装完了（2025-12-11） |
| `train_overlay_zarr.sh` | `ManiFlow/scripts/` | Zarr版訓練スクリプト | ✅ 実装完了（2025-12-11） |

### 既存スクリプト（流用）

| ファイル | 目的 |
|---------|------|
| `HAMSTER/tests/generate_paths_robotwin2_single_vila.py` | 全フレームパス生成（シングルアーム用） |
| `HAMSTER/tests/extract_episode_frames_robotwin2.py` | HDF5からフレーム抽出 |
| `maniflow/policy/maniflow_image_policy.py` | 画像ベースManiFlowポリシー（既存） |
| `maniflow/model/vision_2d/timm_obs_encoder.py` | 画像エンコーダ（既存） |

### 実装順序

1. **パス生成**: ✅ 完了（episode_00: 2025-12-08、全50エピソード: 2025-12-10 進行中 ~36%）
2. **オーバーレイ生成**: ✅ `generate_overlay_images.py` + `overlay_utils.py` 実装完了（2025-12-10）
3. **データセット（PNG版）**: ✅ `robotwin2_overlay_dataset.py` 実装完了（2025-12-10）
4. **設定ファイル（PNG版）**: ✅ 実装完了（2025-12-10）
5. **統合テスト（CPU）**: ✅ Dataset, DataLoader, Policy初期化テスト完了（2025-12-10）
   - R3Mエンコーダ（ResNet-18）: 22.35Mパラメータ
   - DiTXモデル: 55.58Mパラメータ
   - 合計: 77.93Mパラメータ
6. **データローディング問題の発見・解決**: ✅ 完了（2025-12-11）
   - PNG版は1エポック1時間以上 → Zarr版に移行
   - `convert_overlay_to_zarr.py` 実装
   - `robotwin2_overlay_zarr_dataset.py` 実装
   - Zarr版設定ファイル実装
7. **Zarr変換の実行**: ← **次のステップ**
8. **統合テスト（GPU）**: Forward/Backward passテスト
9. **訓練実行**: 本格的な訓練開始

### オーバーレイ画像の仕様（HAMSTER準拠）

**HAMSTERオリジナルの描画方法に完全準拠**（`visualize_hamster_style_comparison.py`の`draw_lines_on_image_cv`関数より）

```python
# パス描画の仕様（HAMSTER準拠）
from matplotlib import cm

# スケーリング（512x512を基準）
scale_factor = max(min(width, height) / 512.0, 1)
circle_radius = int(7 * scale_factor)
circle_thickness = max(1, int(2 * scale_factor))
line_thickness = max(1, int(2 * scale_factor))

# 線の色: jetカラーマップ（青→シアン→緑→黄→赤）
cmap = cm.get_cmap('jet')
colors = (cmap(np.linspace(0, 1, num_subdivisions))[:, :3] * 255).astype(np.uint8)

# パスを100分割して補間し、各セグメントに異なる色を適用
num_subdivisions = 100

# グリッパーマーカー（状態が変化した点のみ描画）:
#   - Open: 青の円 (255, 0, 0) in BGR
#   - Close: 赤の円 (0, 0, 255) in BGR
# マーカーは輪郭のみ（thickness=circle_thickness）
```

**重要**: グリッパーマーカーは「状態が変化した点」にのみ描画される（毎フレームではない）

### RoboTwin 2.0 シングルアームタスク（6タスク）

```python
SINGLE_ARM_TASKS = [
    "beat_block_hammer",
    "click_bell",
    "move_can_pot",
    "open_microwave",
    "place_object_stand",
    "turn_switch",
]
```

### 訓練コマンド（Zarr版・推奨）

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow

# Step 1: Zarr変換（初回のみ、約10-15分）
/usr/bin/singularity exec \
    --bind /gscratch/:/gscratch/:rw \
    --bind /mmfs1/:/mmfs1/:rw \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    python scripts/convert_overlay_to_zarr.py

# Step 2: 訓練実行（Zarr版）
# PYTHONUNBUFFERED=1: tqdmの進捗バーがリアルタイムで表示される（重要！）
/usr/bin/singularity exec --nv \
    --bind /gscratch/:/gscratch/:rw \
    --bind /mmfs1/:/mmfs1/:rw \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONPATH="/mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/r3m" \
    --env WANDB_MODE=disabled \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    python maniflow/workspace/train_maniflow_robotwin_workspace.py \
    --config-name=maniflow_overlay_image_policy_robotwin2_zarr

# または簡単なスクリプト
./scripts/train_overlay_zarr.sh 0 42  # GPU 0, seed 42
```

**重要な環境変数**:
- `PYTHONUNBUFFERED=1`: Pythonの出力バッファリングを無効化し、tqdmの進捗バーをリアルタイムで表示
- `PYTHONNOUSERSITE=1`: ユーザーローカルパッケージを無視（バージョン競合回避）
- `WANDB_MODE=disabled`: Weights & Biasesを無効化（ローカル実行時）

**Zarr変換の出力先**: `/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/robotwin2_overlay_single_arm.zarr`

### 訓練コマンド（PNG版・非推奨、遅い）

```bash
# 注意: PNG版は1エポック1時間以上かかるため非推奨
/usr/bin/singularity exec --nv \
    --bind /gscratch/:/gscratch/:rw \
    --bind /mmfs1/:/mmfs1/:rw \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    python maniflow/workspace/train_maniflow_robotwin_workspace.py \
    --config-name=maniflow_overlay_image_policy_robotwin2
```

### 備考

- RoboTwin 1.0 (`.zarr`) と RoboTwin 2.0 (`.h5`) はデータ形式が異なる
- 既存の `robotwin_image_dataset.py` は zarr 専用のため、HDF5 用に新規作成が必要
- **ManiFlowの画像ポリシーを使用**するため、Point Cloudエンコーダは不要
- Language条件付けを有効にしてマルチタスク対応

---

## ✅ データローディング問題の解決（2025-12-11）

### 問題の発見

PNG版データセット（`robotwin2_overlay_dataset.py`）でManiFlow訓練を実行したところ、**1エポックに1時間以上**かかる問題が発生。本来ManiFlowの訓練は1バッチ数秒で終わるはず。

### ボトルネック分析

| 項目 | 元のManiFlow (zarr) | PNG版データセット |
|------|---------------------|-------------------|
| データ格納 | Zarr (メモリマップ) | 個別PNGファイル |
| 初期化時 | 全データをメモリにロード | ファイル一覧のみ取得 |
| `__getitem__` | NumPy配列スライス（超高速） | **毎回ディスクI/O** |
| アクションデータ | メモリ上のNumPy | **毎回HDF5/ZIPを開く** |

**主な問題点**:

1. **画像ロードが毎回ディスクI/O**: horizon=16で32回の`cv2.imread()`が毎サンプルで発生
2. **HDF5ファイルを毎回開閉**: ファイルオープンのオーバーヘッド
3. **Initial overlayが冗長**: 同じframe 0のオーバーレイを16回読み込む

### 解決策: Zarr形式に事前変換

PNG画像とHDF5アクションを**Zarrファイルに事前変換**することで、元のManiFlowと同じパフォーマンスを実現。

**変換スクリプト**: `ManiFlow/ManiFlow/scripts/convert_overlay_to_zarr.py`

**変換コマンド**:
```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow

# ドライラン（確認のみ）
python scripts/convert_overlay_to_zarr.py --dry-run

# 変換実行
python scripts/convert_overlay_to_zarr.py

# タスク指定
python scripts/convert_overlay_to_zarr.py --tasks beat_block_hammer click_bell move_can_pot
```

**出力先**: `/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/robotwin2_overlay_single_arm.zarr`

**Zarr構造**:
```
robotwin2_overlay_single_arm.zarr/
├── data/
│   ├── overlay_image    # (N, 3, 224, 224) float32
│   ├── action           # (N, 14) float32
│   └── state            # (N, 14) float32
└── meta/
    └── episode_ends     # (num_episodes,) int64
```

### パフォーマンス比較（予想）

| 項目 | PNG版（旧） | Zarr版（新） |
|------|-----------|------------|
| データロード | 毎回ディスクI/O | メモリから読み込み |
| `__getitem__` | ~100ms+ | ~1ms以下 |
| 1エポック | 1時間以上 | 数分 |
| メモリ使用量 | 低い | 高い（~13.5GB） |

### 新規作成ファイル

| ファイル | 説明 |
|---------|------|
| `convert_overlay_to_zarr.py` | PNG+HDF5 → Zarr変換スクリプト |
| `robotwin2_overlay_zarr_dataset.py` | Zarrベースの高速データセット |
| `robotwin2_overlay_single_arm_zarr.yaml` | タスク設定（Zarr版） |
| `maniflow_overlay_image_policy_robotwin2_zarr.yaml` | ポリシー設定（Zarr版） |
| `train_overlay_zarr.sh` | 訓練スクリプト |

### 使用手順

```bash
# Step 1: Zarr変換（初回のみ）
python scripts/convert_overlay_to_zarr.py

# Step 2: 訓練
./scripts/train_overlay_zarr.sh 0 42  # GPU 0, seed 42
```

---

## 代替案: アプローチB（Concat 6ch画像入力）

**状態**: 保留（アプローチAが不十分な場合に実装）

### 概要

アプローチAでパス情報と視覚情報の分離が不十分な場合の代替案。

```
入力: [Initial Concat Image] + [Current Concat Image] + Proprio + Language → ManiFlow
```

### Concat画像の構成

```
Channel 0-2: RGB画像
Channel 3-5: パスのみ画像（黒背景にパス描画）
```

### 必要な変更

- `TimmObsEncoder` の入力層を 3ch → 6ch に変更
- または新規エンコーダ `SixChannelObsEncoder` を作成

---

## ✅ VILAによるパス生成（episode_00のみ、2025-12-08 完了）

### 最終結果（episode_00のみ）

| タスク | フレーム数 | Paths | Raw | 成功率 |
|--------|-----------|-------|-----|--------|
| beat_block_hammer | 126 | 126 | 126 | 100% |
| click_bell | 81 | 67 | 81 | 82.7% |
| move_can_pot | 154 | 154 | 154 | 100% |
| open_microwave | 427 | 119 | 427 | 27.9% |
| place_object_stand | 138 | 132 | 138 | 95.7% |
| turn_switch | 93 | 93 | 93 | 100% |
| **合計** | **1,019** | **691** | **1,019** | **67.8%** |

- **Paths**: パース成功数（`<ans>`タグから座標抽出成功）
- **Raw**: 処理済みフレーム数（モデル出力あり）
- `open_microwave`タスクはパース失敗が多いが、全フレームの処理自体は完了

### 出力先

```
HAMSTER/results/robotwin2_single_6tasks_vila/{task}/episode_00/
├── frames/       # 入力フレーム画像 (.png)
├── paths/        # 生成されたパス (.pkl) - List[(x, y, gripper_state)]
└── raw_outputs/  # モデルの生レスポンス (.txt)
```

---

## ✅ 全50エピソードのフレーム抽出（2025-12-08 完了）

### 抽出結果

| タスク | エピソード数 | フレーム数 |
|--------|-------------|-----------|
| beat_block_hammer | 50 | 5,858 |
| click_bell | 50 | 3,986 |
| move_can_pot | 50 | 7,772 |
| open_microwave | 50 | 24,810 |
| place_object_stand | 50 | 7,140 |
| turn_switch | 50 | 5,006 |
| **合計** | **300** | **54,572** |

**ディスク使用量**: 6.7 GB

### 出力先

```
HAMSTER/results/robotwin2_single_6tasks_vila/{task}/episode_{00-49}/frames/
└── frame_{0000-XXXX}.png
```

---

## 🔄 進行中: 全50エピソードに対するVILAパス生成（2025-12-10更新）

### 概要

- 6タスク × 50エピソード = 300エピソード分のパス生成が必要
- 合計約53,553フレームに対してパス生成を実行
- フレーム抽出は完了済み
- **4GPU並列実行で進行中**（`HAMSTER/tests/parallel_vila/`）

### 現在の進捗（2025-12-10）

- 処理済み: **19,481 / 53,553フレーム（約36%）**
- 4つのVILAサーバー（ポート8000-8003）が稼働中
- 残り: 約34,072フレーム

### 実行コマンド（参考）

```bash
# VILAサーバー起動（別ターミナル）
cd /mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER
./start_vila_server_hyak.sh

# パス生成実行（全50エピソード）
/usr/bin/singularity exec \
    --env PYTHONPATH="/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila" \
    --env SSL_CERT_FILE="" \
    --env SSL_CERT_DIR="" \
    --env REQUESTS_CA_BUNDLE="" \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    /bin/bash -c 'cd /mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/tests && /usr/bin/python generate_paths_robotwin2_single_vila.py --episodes 50'
```

### 推定所要時間

- episode_00（1,019フレーム）: 約30分
- 全50エピソード（54,572フレーム）: 約27時間（推定）

### 注意事項

- GPUノードの確保時間を十分に取る（24時間以上推奨）
- VILAサーバーが起動していることを確認してからパス生成を実行
- 途中で中断した場合、既存のパスはスキップされる（再開可能）

### 複数GPU並列実行（検討中）

4台のGPU（H200/L40s）を使用して並列処理を行うことで、処理時間を約1/4に短縮可能。

**基本方針:**
- 各GPUで別々のVILAサーバーを異なるポート（8000-8003）で起動
- タスクまたはエピソードを分散して処理

```bash
# 例: GPU 0-3 でそれぞれサーバー起動
CUDA_VISIBLE_DEVICES=0 ./start_vila_server_hyak.sh --port 8000
CUDA_VISIBLE_DEVICES=1 ./start_vila_server_hyak.sh --port 8001
# ...
```

**必要な修正:**
- `start_vila_server_hyak.sh` に `--port` オプション追加
- `generate_paths_robotwin2_single_vila.py` に `--port`, `--tasks` オプション追加

**推定所要時間:**
- 1 GPU: 約27時間
- 4 GPU: 約7時間

---

## 🔄 進行中: PEEK (VILA-3B) によるパス生成（2025-12-11更新）

### 概要

HAMSTER（VILA-13B）に加えて、**PEEK（VILA-3B）** でもパス生成を並行して実施中。
PEEKはより軽量なモデルで、Trajectory（パス）とMask（注目領域）を同時に出力する。

### HAMSTERとPEEKの比較

| 項目 | HAMSTER (VILA-13B) | PEEK (VILA-3B) |
|------|-------------------|----------------|
| モデルサイズ | 13B (~26GB VRAM) | 3B (~7GB VRAM) |
| 出力 | Trajectory + Gripper State | Trajectory + Mask |
| ポート | 8000-8003 | 8010-8013 |
| 出力ディレクトリ | `robotwin2_single_6tasks_vila/` | `robotwin2_single_6tasks_peek/` |

### 実行スクリプト

```
HAMSTER/tests/parallel_peek/
├── config.py           # 設定（ポート8010-8013、モデルパス、プロンプト）
├── start_servers.sh    # 4GPU並列でPEEKサーバー起動
├── stop_servers.sh     # サーバー停止
├── generate_paths.py   # 動的タスクキューで並列パス生成
├── logs/               # サーバーログ
└── README.md           # 詳細ドキュメント
```

### 出力構造

```
robotwin2_single_6tasks_peek/
├── {task}/
│   └── episode_{XX}/
│       ├── frames/                     # 入力フレーム
│       ├── paths/                      # Trajectory出力
│       │   └── trajectory_frame_{XXXX}.pkl
│       ├── masks/                      # Mask出力
│       │   └── mask_frame_{XXXX}.pkl
│       └── raw_outputs/                # 生VLMレスポンス
│           └── raw_frame_{XXXX}.txt
```

### 実行手順

```bash
# 1. GPUノード確保（4GPU）
srun -p gpu-l40s -A escience --nodes=1 --cpus-per-task=32 --mem=400G --time=24:00:00 --gpus=4 --pty /bin/bash

# 2. サーバー起動（4GPU並列）
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/tests/parallel_peek
./start_servers.sh

# 3. パス生成実行
/usr/bin/singularity exec \
    --bind /gscratch/:/gscratch/:rw \
    --bind /mmfs1/:/mmfs1/:rw \
    --env PYTHONPATH="/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila" \
    --env SSL_CERT_FILE="" \
    --env SSL_CERT_DIR="" \
    --env REQUESTS_CA_BUNDLE="" \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    python generate_paths.py --episodes 50

# 4. サーバー停止
./stop_servers.sh
```

### 現在の進捗（2025-12-11）

- 処理済み: **約2,900 / 53,553フレーム（約5%）**
- 4つのPEEKサーバー（ポート8010-8013）が稼働中
- スキップ条件: `raw_frame_*.txt` が存在すればスキップ（途中再開可能）

### 備考

- 現時点ではMask情報は使用せず、Trajectoryのみを使用予定
- PEEKはgripper_stateを出力しないため、オーバーレイ画像生成時はマーカーなしで描画

---

## 環境構築・実行手順（Hyak HPC）

### 前提条件

- Hyak HPCのGPUノードにアクセス可能であること
- H200/A40等のGPU（26GB以上のVRAM）

### 1. GPUノードの確保

```bash
srun -p gpu-a40 -A escience --nodes=1 --cpus-per-task=32 --mem=400G --time=24:00:00 --gpus=1 --pty /bin/bash
```

### 2. 必要なファイル・ディレクトリ

| パス | 説明 |
|------|------|
| `/gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif` | Singularityコンテナ（PyTorch 2.5 + CUDA 12.1） |
| `/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila/` | VILA用外部パッケージディレクトリ |
| `HAMSTER/Hamster_dev/VILA1.5-13b-...` | HAMSTERファインチューン済みVILAモデル |
| `HAMSTER/VILA/` | VILAリポジトリ（llavaパッケージ） |

### 3. 外部パッケージのインストール（初回のみ）

**重要**: `vila.sif`（PyTorch 2.3）ではなく`hamster-maniflow_latest.sif`（PyTorch 2.5）を使用する。
H200/CUDA 13.0ドライバとの互換性のため、PyTorch 2.5が必要。

```bash
# site-packages-vila ディレクトリを作成（まだない場合）
mkdir -p /gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila

# 必要なパッケージをインストール
/usr/bin/singularity exec \
    --bind /gscratch/:/gscratch/:rw \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    /bin/bash -c '/usr/bin/python -m pip install --target=/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila datasets'

/usr/bin/singularity exec \
    --bind /gscratch/:/gscratch/:rw \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    /bin/bash -c '/usr/bin/python -m pip install --target=/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila "huggingface-hub>=0.34.0,<1.0" "transformers>=4.40.0,<4.50.0"'

/usr/bin/singularity exec \
    --bind /gscratch/:/gscratch/:rw \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    /bin/bash -c '/usr/bin/python -m pip install --target=/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila openai httpx'
```

### 4. VILAサーバーの起動

```bash
cd /mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER
./start_vila_server_hyak.sh
```

サーバーが正常に起動すると以下のようなメッセージが表示される：
```
==========================================
VILA Server (HAMSTER finetuned)
==========================================
Model: .../VILA1.5-13b-robopoint_1432k+rlbench_all_tasks_256_1000_eps_sketch_v5_alpha+...
Port: 8000
...
Loading checkpoint shards: 100%|██████████| 6/6 [00:XX<00:00, ...]
Model VILA1.5-13b-... loaded successfully. Context length: 2048
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**ポイント**:
- モデルロードに約60秒かかる
- 約26GB VRAMを使用
- ポート8000でリッスン

### 5. パス生成の実行

**サーバーとは別のターミナルで実行**（または同じノードでバックグラウンド実行）

```bash
/usr/bin/singularity exec \
    --env PYTHONPATH="/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila" \
    --env SSL_CERT_FILE="" \
    --env SSL_CERT_DIR="" \
    --env REQUESTS_CA_BUNDLE="" \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    /bin/bash -c 'cd /mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/tests && /usr/bin/python generate_paths_robotwin2_single_vila.py --episodes 1'
```

**重要な環境変数**:
- `PYTHONPATH`: site-packages-vilaを優先読み込み
- `SSL_CERT_FILE=""`: SSL証明書エラー回避（OpenAIクライアント用）

### 6. 進捗確認

```bash
# 各タスクのパス生成数を確認
B="/mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/robotwin2_single_6tasks_vila"
for task in beat_block_hammer click_bell move_can_pot open_microwave place_object_stand turn_switch; do
    echo "$task: paths=$(ls $B/$task/episode_00/paths 2>/dev/null | wc -l) raw=$(ls $B/$task/episode_00/raw_outputs 2>/dev/null | wc -l)"
done
```

---

## 起動スクリプトの詳細（start_vila_server_hyak.sh）

```bash
#!/bin/bash
# Singularity image path (using hamster-maniflow for PyTorch 2.5 + CUDA 13.0 compatibility)
SIF_PATH="/gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif"

singularity exec --nv \
    --bind /gscratch/:/gscratch/:rw \
    --env PYTHONPATH="/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila:$SCRIPT_DIR/VILA" \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HOME="/gscratch/scrubbed/naoto03/.cache/huggingface" \
    --env TRANSFORMERS_CACHE="/gscratch/scrubbed/naoto03/.cache/huggingface" \
    "$SIF_PATH" \
    python -W ignore "$SCRIPT_DIR/server.py" \
        --port 8000 \
        --model-path "$MODEL_PATH" \
        --conv-mode vicuna_v1
```

**重要なポイント**:
- `--nv`: NVIDIA GPUを有効化
- `PYTHONNOUSERSITE=1`: ユーザーローカルパッケージを無視（バージョン競合回避）
- `site-packages-vila`を`PYTHONPATH`の先頭に配置

---

## トラブルシューティング

### 問題1: CUDA driver error: invalid argument

```
RuntimeError: CUDA driver error: invalid argument
```

**原因**: PyTorch 2.3の`tile()`カーネルがCUDA 13.0ドライバと非互換

**解決策**: `vila.sif`ではなく`hamster-maniflow_latest.sif`（PyTorch 2.5）を使用

### 問題2: SSL証明書エラー

```
FileNotFoundError: [Errno 2] No such file or directory (SSL_CERT_FILE)
```

**解決策**: 環境変数をクリア
```bash
--env SSL_CERT_FILE="" \
--env SSL_CERT_DIR="" \
--env REQUESTS_CA_BUNDLE=""
```

### 問題3: OpenAIクライアントの`proxies`引数エラー

```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

**解決策**: 新しいバージョンのopenaiパッケージをインストール
```bash
pip install --target=.../site-packages-vila openai httpx
```

### 問題4: huggingface-hub/transformersバージョン不整合

```
HFValidationError: Repo id must be in the form 'repo_name'
```

**解決策**: 互換性のあるバージョンをインストール
```bash
pip install --target=... "huggingface-hub>=0.34.0,<1.0" "transformers>=4.40.0,<4.50.0"
```

### 問題5: R3Mモジュールが見つからない

```
ModuleNotFoundError: No module named 'r3m'
```

**解決策**: ManiFlowの`third_party/r3m`をPYTHONPATHに追加
```bash
--env PYTHONPATH="/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/r3m"
```

### 問題6: R3M重みダウンロード時のディスク容量不足

```
OSError: [Errno 122] Disk quota exceeded
```

**原因**: R3Mがデフォルトで`~/.r3m/`に重みを保存しようとするが、ホームディレクトリのクォータが不足

**解決策**: `ManiFlow/third_party/r3m/r3m/__init__.py`の`load_r3m`関数を修正
```python
# 変更前
home = os.path.join(expanduser("~"), ".r3m")

# 変更後
home = os.path.join("/gscratch/scrubbed/naoto03", ".r3m")
```

**注**: R3M-ResNet18の重みは約330MB。`/gscratch/scrubbed/naoto03/.r3m/r3m_18/`に保存される。

---

## ✅ 見かけ上のデッドロック問題（解決済み・2025-12-13）

### 問題の概要

ManiFlow訓練中に「デッドロック」と思われた現象が複数回発生したが、**実際にはデッドロックではなく、tqdm出力のバッファリング問題**であることが判明。

### 見かけ上のデッドロック #1: Epoch 1 batch 88-89（解決済み）

**症状**:
- Epoch 1 batch 88-89付近で訓練が停止したように見えた
- GPU使用率: 0%（GPUメモリは使用中）
- プロセス状態: Running

**原因**:
- `num_workers=8`設定でDataLoaderの並列処理が問題を起こしていた
- `robotwin2_overlay_zarr_dataset.py`の`_sample_to_data`メソッドで`initial_overlay`を取得する際にランダムアクセスが発生
- 複数ワーカーがZarrデータの同じインデックスに同時アクセス

**解決策**:
```yaml
# maniflow_overlay_image_policy_robotwin2_zarr.yaml
dataloader:
  num_workers: 0  # ワーカーを無効化
  persistent_workers: false  # num_workers=0の場合は必須false
```

### 見かけ上のデッドロック #2: Epoch 2 batch 178（誤診断・実際は正常）

**症状**:
- Epoch 2 batch 178/267で訓練が停止したように見えた
- GPU使用率: 0%（nvidia-smiで確認した時点で）
- tqdmの進捗バーが更新されない

**当初の調査**:
- `num_workers=0`は既に設定済み
- `pin_memory=true`が原因かと疑った

**真の原因（2025-12-13判明）**:
- **tqdmの出力バッファリング**により進捗表示が更新されなかった
- 訓練自体は正常に進行していた
- Singularityコンテナ内でのstdout/stderrバッファリングが原因

**解決策**:
デバッグログをファイルに直接書き出すことで、訓練が実際に進行していることを確認：
```python
# 訓練ループにデバッグログを追加
def debug_log(msg):
    with open(debug_log_path, 'a') as f:
        f.write(f"[{timestamp}] {msg}\n")
        f.flush()  # 即座にフラッシュ
```

**結論**:
- `pin_memory: true` は問題なし
- `num_workers: 0` のみが必要な設定変更
- デバッグログで訓練がEpoch 2 batch 178を超えて正常に進行していることを確認

### 最終的な設定

```yaml
# maniflow_overlay_image_policy_robotwin2_zarr.yaml
dataloader:
  batch_size: 64
  num_workers: 0  # 必須：ランダムアクセスパターンのため
  shuffle: true
  pin_memory: true  # 問題なし
  persistent_workers: false  # num_workers=0のため

val_dataloader:
  batch_size: 64
  num_workers: 0
  shuffle: false
  pin_memory: true
  persistent_workers: false
```

### 教訓

1. **tqdm出力のバッファリング**: Singularityコンテナ内ではstdout/stderrがバッファリングされることがある
2. **進捗確認の方法**: tqdmではなく、ファイルへの直接ログ出力（flush付き）が確実
3. **GPU使用率0%の解釈**: バッチ間の短い待機時間にnvidia-smiを実行すると0%に見える場合がある

---

## 関連ファイル一覧

| ファイル | パス | 説明 |
|---------|------|------|
| 起動スクリプト | `HAMSTER/start_vila_server_hyak.sh` | VILAサーバー起動用 |
| サーバー本体 | `HAMSTER/server.py` | FastAPIサーバー (port 8000) |
| パス生成スクリプト | `HAMSTER/tests/generate_paths_robotwin2_single_vila.py` | 6タスク用パス生成 |
| 結果出力先 | `HAMSTER/results/robotwin2_single_6tasks_vila/` | 生成されたパスの保存先 |
| Singularityコンテナ | `/gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif` | PyTorch 2.5 + CUDA 12.1 |
| 外部パッケージ | `/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila/` | datasets, openai等 |
| VILAモデル | `HAMSTER/Hamster_dev/VILA1.5-13b-...` | HAMSTERファインチューン済み |
| VILAリポジトリ | `HAMSTER/VILA/` | llavaパッケージ |
