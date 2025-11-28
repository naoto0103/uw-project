# HAMSTER-ManiFlow統合プロジェクト: 進捗状況

**最終更新**: 2025-11-20
**プロジェクトディレクトリ**: `~/HAMSTER-ManiFlow-Integration/`

---

## 📊 プロジェクト概要

### 目的
HAMSTERの高レベル経路計画（VLMベース）とManiFlowの低レベル精密制御（点群ベース）を統合し、階層的なロボット操作システムを構築する。

### 科学的根拠
HAMSTERの論文（arXiv 2502.05485, Table 3）のアブレーション研究により、**経路情報を別次元として入力する方法**が画像オーバーレイよりも優れていることが実証されている：

| 手法 | 成功率 |
|------|--------|
| 画像オーバーレイ | 0.83 |
| **別次元入力（連結）** | **1.00 (+17%)** |

### 採用アプローチ
点群データと経路トークンを独立した特徴量として扱い、Attention機構で統合する。

---

## 🖥️ 開発環境

### ハードウェア
- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **CUDA Driver**: 13.0

### ソフトウェア要件

**重要**: RTX 5090はCUDA 12.8+を必要とするため、PyTorch nightlyビルドが必須です。

#### Conda環境構成（3環境）

1. **`vila`環境** (HAMSTER用)
   ```bash
   conda activate vila
   ```
   - Python: 3.10.19
   - **PyTorch: 2.10.0.dev20251114+cu128** (nightly)
   - 用途: HAMSTERサーバー（VILA-1.5-13B VLM）
   - 主要パッケージ: transformers, accelerate, gradio, openai

2. **`qwen3`環境** (Qwen3-VL用) ⭐新規
   ```bash
   conda activate qwen3
   ```
   - Python: 3.10.x
   - **PyTorch: 2.10.0.dev20251114+cu128** (nightly)
   - **transformers: >= 4.57.0**
   - 用途: Qwen3-VLサーバー（Qwen3-VL-8B-Instruct VLM）
   - 主要パッケージ: transformers, fastapi, uvicorn, pillow, openai

3. **`maniflow`環境** (ManiFlow用)
   ```bash
   conda activate maniflow
   ```
   - Python: 3.10.x
   - **PyTorch: 2.10.0.dev20251114+cu128** (nightly)
   - **pytorch3d: 0.7.8** (ソースからビルド)
   - 用途: ManiFlowポリシー訓練・推論
   - 主要パッケージ: hydra-core, zarr, wandb

#### PyTorch Nightly インストール

RTX 5090対応のため、両環境でPyTorch nightlyを使用:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### pytorch3d ビルド要件

**重要**: pytorch3d 0.7.8はPyTorch nightlyと互換性を持たせるため、**gcc 11.2.0**でビルドする必要があります。

```bash
# gcc 15.2.0ではC++ヘッダ競合が発生するため、gcc 11.2.0を使用
export CC=gcc-11
export CXX=g++-11
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"
```

**理由**: gcc 15.2.0のC++標準ライブラリヘッダ（`<cstdint>`など）がPyTorch nightlyのヘッダと競合するため。

---

## ✅ 完了したタスク

### Phase 0: プロジェクトセットアップと調査 (完了)

#### 1. プロジェクト構造の作成 ✓
**実施日**: 2025-11-12
**場所**: `/home/naoto/HAMSTER-ManiFlow-Integration/`

```bash
HAMSTER-ManiFlow-Integration/
├── HAMSTER/          # 4.9MB - HAMSTERリポジトリ
├── ManiFlow/         # 217MB - ManiFlowリポジトリ
├── IMPLEMENTATION_PLAN.md  # 実装計画書（59KB）
└── PROJECT_PROGRESS.md     # このファイル
```

**状態**: ✅ 完了

---

#### 2. HAMSTERリポジトリのクローン ✓
**実施日**: 2025-11-12
**リポジトリ**: https://github.com/liyi14/HAMSTER_beta

**クローンされたファイル**:
```
HAMSTER/
├── server.py                    # FastAPI VLMサーバー
├── gradio_server_example.py     # Gradioデモインターフェース
├── setup_server.sh              # サーバー起動スクリプト
├── requirements.txt             # 依存関係
├── README.md                    # セットアップ手順
└── examples/                    # サンプル画像
```

**主要な発見**:
- HAMSTERはFastAPIベースのRESTful APIサーバーとして実装
- VILA-1.5-13B VLMを使用（HuggingFace: `yili18/Hamster_dev`）
- 入力: RGB画像 + 自然言語タスク記述
- 出力: 2D正規化座標列 + グリッパアクション

**状態**: ✅ 完了

---

#### 3. ManiFlowリポジトリのクローン ✓
**実施日**: 2025-11-12
**リポジトリ**: https://github.com/geyan21/ManiFlow_Policy

**クローンされた主要コンポーネント**:
```
ManiFlow/
├── ManiFlow/maniflow/
│   ├── policy/
│   │   ├── base_policy.py
│   │   ├── maniflow_pointcloud_policy.py  # 3Dポリシー
│   │   └── maniflow_image_policy.py       # 2Dポリシー
│   ├── model/
│   │   ├── vision_3d/
│   │   │   └── pointnet_extractor.py      # DP3Encoder
│   │   ├── vision_2d/
│   │   │   └── timm_obs_encoder.py        # TimmObsEncoder
│   │   └── diffusion/
│   │       └── ditx.py                    # DiTX Transformer
│   ├── dataset/
│   │   ├── robotwin_dataset.py            # 点群データセット
│   │   └── robotwin_image_dataset.py      # 画像データセット
│   └── config/                            # YAML設定ファイル
├── scripts/
│   ├── train_eval_robotwin.sh
│   ├── train_eval_dex.sh
│   └── gen_demonstrations_robotwin1.0.sh
└── third_party/                           # 外部依存
    ├── RoboTwin1.0/
    ├── Metaworld/
    ├── pytorch3d/
    └── ...
```

**主要な発見**:
- ManiFlowは2D画像と3D点群の両方をサポート
- DiTX (Diffusion Transformer with Cross-Attention)を使用
- Consistency Flow Matching: 1-2ステップで推論可能
- モジュール化された設計で拡張が容易

**状態**: ✅ 完了

---

#### 4. システムアーキテクチャの詳細調査 ✓
**実施日**: 2025-11-12

**調査したファイル**:
- ✅ `/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/server.py` (259行)
  - FastAPIエンドポイント: `/chat/completions`
  - VILA VLMのロード・推論ロジック
  - 画像のbase64エンコーディング処理

- ✅ `/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/gradio_server_example.py` (230行)
  - 経路描画関数: `draw_lines_on_image_cv()` (55-120行目)
  - レスポンスパーサー: `process_answer()` (122-139行目)
  - OpenAI API互換のクライアント実装

- ✅ `/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/maniflow/policy/base_policy.py` (26行)
  - ベースインターフェース定義
  - `predict_action()`: 観測 → アクション
  - `set_normalizer()`: 正規化の設定

- ✅ `/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/maniflow/policy/maniflow_pointcloud_policy.py` (200行+)
  - `ManiFlowTransformerPointcloudPolicy`クラス
  - DP3Encoderの使用（点群エンコーディング）
  - DiTXモデルとConsistency Flow統合
  - トレーニング・推論ロジック

- ✅ `/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/maniflow/policy/maniflow_image_policy.py` (200行+)
  - `ManiFlowTransformerImagePolicy`クラス
  - TimmObsEncoderの使用（画像エンコーディング）
  - 2D画像ベースのポリシー

- ✅ `/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/maniflow/model/vision_3d/pointnet_extractor.py` (344行)
  - `PointNetEncoderXYZ`: 3チャンネル（XYZ）点群エンコーダ
  - `PointNetEncoderXYZRGB`: 6チャンネル（XYZRGB）点群エンコーダ
  - `DP3Encoder`: 点群 + 状態 → 統合特徴 (209-305行目)
    - 入力: `point_cloud`, `agent_pos`, (オプション) `imagin_robot`
    - 出力: 統合特徴ベクトル

- ✅ `/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/maniflow/model/vision_2d/timm_obs_encoder.py` (100行+)
  - `TimmObsEncoder`: timm/R3Mベースの画像エンコーダ
  - マルチカメラ対応
  - Data augmentation統合

**データフォーマットの理解**:

**ManiFlow点群ポリシーの入力**:
```python
obs_dict = {
    'point_cloud': torch.Tensor,  # [B, To, N, 3/6]
    'agent_pos': torch.Tensor,    # [B, To, D]
}
```

**ManiFlow画像ポリシーの入力**:
```python
obs_dict = {
    'head_cam': torch.Tensor,     # [B, To, 3, H, W]
    'agent_pos': torch.Tensor,    # [B, To, D]
}
```

**HAMSTERの出力**:
```python
path_points = [
    (x1, y1),  # 正規化座標 [0, 1]
    (x2, y2),
    # ... グリッパアクション
    # <action>Close Gripper</action> → (1000.0, 1000.0)
    # <action>Open Gripper</action> → (1001.0, 1001.0)
]
```

**状態**: ✅ 完了

---

#### 5. 設定ファイルの分析 ✓
**実施日**: 2025-11-12

**調査した設定**:

**点群ポリシー設定** (`maniflow_pointcloud_policy_robotwin.yaml`):
```yaml
shape_meta:
  obs:
    point_cloud:
      shape: [1024, 6]  # N点, XYZRGB
      type: point_cloud
    agent_pos:
      shape: [14]
      type: low_dim
  action:
    shape: [14]

policy:
  encoder_type: "DP3Encoder"
  encoder_output_dim: 128
  use_pc_color: false
  pointnet_type: "pointnet"
  downsample_points: true
  pointcloud_encoder_cfg:
    num_points: 128  # visual_cond_len
    pointwise: true
```

**画像ポリシー設定** (`maniflow_image_timm_policy_robotwin.yaml`):
```yaml
shape_meta:
  obs:
    head_cam:
      shape: [3, 224, 224]
      type: rgb
    agent_pos:
      shape: [14]
      type: low_dim

policy:
  obs_encoder:
    _target_: maniflow.model.vision_2d.timm_obs_encoder.TimmObsEncoder
    model_name: 'r3m'
    pretrained: False
```

**主要パラメータ**:
- `horizon`: 16 (アクション予測の時間長)
- `n_obs_steps`: 2 (観測履歴の長さ)
- `n_action_steps`: 16 (実行するアクション数)
- `num_inference_steps`: 10 → **2** (Consistency Flowで削減可能)
- `batch_size`: 256 (点群), 128 (画像)

**状態**: ✅ 完了

---

#### 6. データセット構造の理解 ✓
**実施日**: 2025-11-12

**調査したファイル**:
- ✅ `ManiFlow/maniflow/dataset/robotwin_dataset.py`
- ✅ `ManiFlow/maniflow/dataset/robotwin_image_dataset.py`

**データセットフォーマット (.zarr)**:
```python
# 点群データセット
buffer_keys = [
    'point_cloud',  # [N_episodes, T, 1024, 6]
    'state',        # [N_episodes, T, 14]
    'action',       # [N_episodes, T, 14]
]

# 画像データセット
buffer_keys = [
    'head_camera',  # [N_episodes, T, 3, 224, 224]
    'state',        # [N_episodes, T, 14]
    'action',       # [N_episodes, T, 14]
]
```

**データ拡張**:
- 点群: Color Jitter (brightness, contrast, saturation, hue)
- 画像: RandomCrop, RandomRotation, ColorJitter

**状態**: ✅ 完了

---

#### 7. HAMSTER論文の詳細調査 ✓
**実施日**: 2025-11-12
**論文**: arXiv 2502.05485

**重要な発見**:

**Table 3: View Invarianceアブレーション研究**
| Method | Original Camera | Novel Camera |
|--------|-----------------|--------------|
| HAMSTER+RVT2 (Overlay) | 0.83 | 0.73 |
| **HAMSTER+RVT2 (Concat)** | **1.00** | **0.98** |

**主要な知見**:
1. **別次元入力（Concat）が優位**:
   - オリジナル視点: +17ポイント (0.83 → 1.00)
   - 新規視点: +25ポイント (0.73 → 0.98)

2. **理由（論文より）**:
   > "RVT2's virtual reprojection can fragment the 2D path when it is directly drawn on the image, making it harder for RVT2 to decode."

   仮想再投影により画像オーバーレイされた経路が断片化され、デコードが困難になる。

3. **制約**:
   > "Less general and not compatible with 3D-DA as it uses a pre-trained CLIP image encoder expecting 3 input channels."

   3D-DAは事前学習済みCLIPエンコーダ（3チャンネル入力）を使用するため非互換。

**統合への示唆**:
- ManiFlowの点群ポリシーでは、経路を独立した特徴トークンとして扱うべき
- 画像ベースの場合、6チャンネル連結が有効
- HAMSTERの実験結果を直接活用できる

**状態**: ✅ 完了

---

#### 8. 統合アーキテクチャの設計 ✓
**実施日**: 2025-11-12

**決定事項**:

1. **入力モダリティ**: 3D点群ポリシーを採用
   - 理由: 3D空間での精密制御に適している
   - HAMSTERの2D経路を3D空間にマッピング可能

2. **経路統合方法**: 独立トークンエンコーディング
   - 経路点を独立した特徴として処理
   - Attention機構で点群特徴と統合
   - HAMSTERの実験結果（別次元入力が優位）に基づく

3. **アーキテクチャ**:
   ```
   入力:
     - point_cloud: [B, N, 3]
     - agent_pos: [B, 14]
     - hamster_path: [B, M, 3]  ← NEW!

   エンコーダ:
     ├─ PointNet → [B, N, 256]
     ├─ StateMLP → [B, 64]
     └─ PathTokenEncoder → [B, 256]  ← NEW!

   統合: [B, N, 576] or [B, 576]

   DiTX → Consistency Flow → Actions
   ```

**状態**: ✅ 完了

---

#### 9. 実装計画書の作成 ✓
**実施日**: 2025-11-12
**ファイル**: `/home/naoto/HAMSTER-ManiFlow-Integration/IMPLEMENTATION_PLAN.md` (59KB)

**内容**:
- ✅ プロジェクト概要と科学的根拠
- ✅ 詳細アーキテクチャ設計
- ✅ 6つの実装フェーズ（Week 1-8）
- ✅ 完全な実装コード例:
  - `PathTokenEncoder` (150行)
  - `HAMSTERPathDP3Encoder` (200行)
  - `HAMSTERRoboTwinDataset` (300行)
  - `HAMSTERManiFlowPolicy` (100行)
- ✅ 設定ファイルテンプレート
- ✅ トレーニング手順
- ✅ 評価・ベンチマーク計画
- ✅ トラブルシューティングガイド

**状態**: ✅ 完了

---

## 🚧 進行中のタスク

Phase 2が完了し、Phase 3（エンコーダ拡張）の準備が整いました。

---

## ✅ 完了したタスク（Phase 1）

### Phase 1: 環境セットアップとHAMSTER統合 ✅ **完了**
**完了日**: 2025-11-15

#### タスク:
- [x] HAMSTERサーバーのセットアップ
  - [x] VILAリポジトリのクローン
  - [x] VILA環境の構築 (`conda activate vila`)
  - [x] **RTX 5090対応**: PyTorch nightly (2.10.0.dev20251114+cu128) インストール
  - [x] **FlashAttention → SDPA フォールバック**: 正常動作確認
  - [x] モデルチェックポイントのダウンロード (26GB)
  - [x] サーバー起動テスト成功

- [x] HAMSTER APIクライアントの実装
  - [x] `test_api_client.py`の作成（`utils/hamster_client.py`の代替）
  - [x] 接続テストスクリプト
  - [x] レスポンスパーサーの実装 (`process_answer()`)
  - [x] 経路描画機能 (`draw_path_on_image()`)

- [x] **2D経路生成テスト成功**
  - [x] non_prehensile.jpg: 2点経路生成 ✓
  - [x] spatial_world_knowledge.jpg: 4点経路生成 ✓
  - [x] 結果画像の保存 ✓

**環境情報**:
- GPU: NVIDIA GeForce RTX 5090 (32GB VRAM)
- CUDA Driver: 13.0
- PyTorch: 2.10.0.dev20251114+cu128 (nightly)
- Python: 3.10.19
- モデル: VILA1.5-13B (Hamster_dev)

**成果物**:
- `test_api_client.py`: 完全な経路生成クライアント
- `setup_server.sh`: 自動モデルパス検出機能追加
- `output_*.jpg`: テスト結果画像

**依存関係**:
- VILAリポジトリ ✅
- HuggingFace モデル: `yili18/Hamster_dev` ✅

**実際の作業時間**: Phase 0から継続、環境構築含め約8時間

---

## ✅ Phase 2: データセット拡張 [完了]

**ステータス**: ✅ 完了
**日付**: 2025年11月15日
**実際の作業時間**: 約6時間

### 実装内容:

1. **HAMSTERRoboTwinDatasetの実装** ✅
   - `ManiFlow/maniflow/dataset/hamster_robotwin_dataset.py` (365行)
   - パス読み込み・キャッシュ機構
   - パディング/切り詰めロジック（固定長50ポイント）
   - RoboTwinDatasetを拡張、HAMSTER 2Dパスを追加
   - 正規化器（Identity normalizer for paths）

2. **バッチ経路生成スクリプト** ✅
   - `scripts/generate_hamster_paths.py` (503行)
   - HAMSTERサーバーAPI統合
   - `.zarr`からの画像抽出
   - 応答パース（座標とグリッパー状態）
   - 進捗保存・再開機能

3. **RoboTwinデータセットダウンロード** ✅
   - Google Driveから6.93GBダウンロード
   - 6タスク: pick_apple_messy, diverse_bottles_pick, dual_bottles_pick_hard, empty_cup_place, block_hammer_beat, shoe_place
   - `data/{task_name}_50.zarr`形式で配置

4. **全データセットへのHAMSTERパス生成** ✅
   - pick_apple_messy: 50/50 成功
   - diverse_bottles_pick: 40/50 成功
   - dual_bottles_pick_hard: 46/50 成功
   - empty_cup_place: 46/50 成功
   - block_hammer_beat: 50/50 成功
   - shoe_place: 49/50 成功
   - **合計: 281/300 エピソード (93.7%成功率)**

5. **テスト** ✅
   - `scripts/test_hamster_core.py` - コア機能テスト
   - `scripts/test_hamster_dataset.py` - 統合テスト
   - 合成テストデータ作成・検証

### 生成されたファイル:
```
data/pick_apple_messy_50/hamster_paths.pkl (6.1KB)
data/diverse_bottles_pick_50/hamster_paths.pkl (11.2KB)
data/dual_bottles_pick_hard_50/hamster_paths.pkl (13.6KB)
data/empty_cup_place_50/hamster_paths.pkl (3.6KB)
data/block_hammer_beat_50/hamster_paths.pkl (4.3KB)
data/shoe_place_50/hamster_paths.pkl (3.7KB)
data/synthetic_test_50/hamster_paths.pkl (0.9KB)
```

### 解決した技術的課題:
- PyTorch nightly (2.10.0.dev+cu128) とpytorch3d 0.7.8の互換性
- gcc 15.2.0のC++ヘッダ競合をgcc 11.2.0で解決
- SequenceSamplerのエピソードインデックスマッピング

---

## 📝 未実装のタスク

---

### Phase 3: エンコーダ拡張 ✅ 完了 (2025-11-16)

#### タスク:
- [x] `PathTokenEncoder`の実装
  - [x] トークン埋め込み層 (3 → 128次元)
  - [x] 位置エンコーディング (学習可能)
  - [x] Attention集約機構 (Transformer Encoder + Query Token)

- [x] `HAMSTERPathDP3Encoder`の実装
  - [x] DP3Encoderの拡張
  - [x] 経路エンコーダの統合
  - [x] 特徴結合ロジック (128 + 64 + 256 = 448次元)

- [x] 単体テスト
  - [x] `tests/test_hamster_integration.py` (420行)
  - [x] グラデーション伝播確認 (全5テスト成功)

- [x] ポリシー統合
  - [x] `maniflow_pointcloud_policy.py`修正
  - [x] `encoder_type="HAMSTERPathDP3Encoder"`選択肢追加

- [x] 設定ファイル作成
  - [x] `config/hamster_maniflow_pointcloud_policy_robotwin.yaml`
  - [x] `config/robotwin_task/pick_apple_messy_hamster.yaml`

**依存関係**: なし（独立して実装可能）

**実際の作業時間**: 約3時間

**主要成果物**:
- `maniflow/model/vision_3d/hamster_path_encoder.py` (558行)
- モデルパラメータ数: 5.26M (obs_encoder: 534K, model: 4.73M)

---

### Phase 3.5: Qwen3-VL統合とゼロショット評価 (✅ 完了: 2025-11-17)

#### 目的:
VLM部分をVILA-1.5-13BからQwen3-VL-8B-Instructに置き換え、ゼロショットでの2Dパス生成能力を評価する。

#### タスク:
- [x] Qwen3-VL環境構築
  - [x] conda環境`qwen3`作成 (Python 3.10)
  - [x] PyTorch 2.10.0.dev+cu128 (nightly, RTX 5090対応)
  - [x] transformers 4.57.1インストール
  - [x] Qwen3-VLリポジトリクローン
- [x] OpenAI互換サーバー実装
  - [x] `server_qwen3.py` 作成 (300行)
  - [x] `/v1/chat/completions` エンドポイント実装
  - [x] `setup_qwen3_server.sh` 作成
  - [x] サーバー起動テスト成功 (port 8001)
- [x] ゼロショット評価
  - [x] Qwen3-VL cookbookの2d_grounding.ipynb分析
  - [x] 2種類のプロンプト方式でテスト
    - [x] HAMSTER形式プロンプト (`<ans>タグ`)
    - [x] Qwen3推奨JSON形式プロンプト (`point_2d`, `gripper`)
  - [x] pick_apple_messyタスクでパス生成
  - [x] VILA vs Qwen3の性能比較
  - [x] 視覚化とエラー分析
- [x] ファイル整理
  - [x] `tests/`ディレクトリ作成
  - [x] `results/`ディレクトリ作成
  - [x] テストスクリプトと結果の整理

#### 実装ファイル:
```
HAMSTER/
├── tests/                              # テストスクリプト
│   ├── test_qwen3_path.py             # オリジナルプロンプトテスト
│   ├── test_qwen3_optimized.py        # 最適化プロンプトテスト
│   ├── compare_all_paths.py           # 3モデル比較分析
│   ├── visualize_comparison.py        # 3モデル視覚化
│   └── visualize_vila_vs_qwen3.py     # VILA vs Qwen3比較
├── results/                            # 実験結果
│   ├── qwen3_test_path.pkl            # オリジナル結果
│   ├── qwen3_optimized_path.pkl       # 最適化結果
│   └── visualizations/                # 比較画像
│       ├── comparison_vila_vs_qwen3_optimized.png
│       ├── comparison_horizontal.png
│       └── ...
├── VILA/                               # 既存
├── Qwen3-VL/                           # クローン済み
├── Hamster_dev/                        # VILA-1.5モデル (51GB)
├── server.py                           # VILAサーバー (port 8000)
├── server_qwen3.py                     # Qwen3サーバー (port 8001)
├── setup_server.sh                     # VILA起動スクリプト
└── setup_qwen3_server.sh               # Qwen3起動スクリプト
```

#### 評価結果: **ゼロショットでは不十分**

**タスク**: "Pick up the apple from the messy table"

| モデル | プロンプト | ウェイポイント数 | ユニーク位置 | パス長 | 座標系 |
|--------|-----------|----------------|-------------|--------|--------|
| **VILA-1.5-13B** | HAMSTER標準 | 4 | 4 | 0.4725 | [0, 1] |
| **Qwen3-VL-8B** | HAMSTER形式 | 2 | 1 | 0.0000 | [0, 1] |
| **Qwen3-VL-8B** | Qwen3推奨JSON | 2 | 1 | 0.0000 | [0, 1000]→[0, 1] |

**Qwen3生成パス例 (最適化プロンプト)**:
```json
[
    {"point_2d": [126, 500], "gripper": "close"},
    {"point_2d": [126, 500], "gripper": "open"}
]
```
→ 変換後: `[[0.126, 0.5, CLOSE], [0.126, 0.5, OPEN]]`

**問題点**:
- ❌ **移動なし**: 両ウェイポイントが同じ座標 (0.126, 0.5)
- ❌ **軌道生成失敗**: その場でグリッパー開閉のみ
- ❌ **タスク理解不足**: "Pick up the apple"の動作を理解していない
- ✅ **出力形式**: JSON形式の出力は正しい
- ✅ **座標範囲**: [0, 1000]を正しく使用（変換も成功）

**主要な発見**:
1. **ゼロショット性能の限界**: Qwen3-VLは2D grounding能力は高い（RefCOCO 82-87%）が、ロボット軌道生成タスクには不十分
2. **プロンプト最適化の効果なし**: HAMSTER形式とQwen3推奨JSON形式の両方で同様の結果（2ウェイポイント、移動なし）
3. **ファインチューニングの重要性**: VILAの1.2Mロボットデータでの学習が軌道生成に不可欠
4. **座標系の違い**: Qwen3-VLは[0, 1000]、VILAは[0, 1]（変換ロジック実装済み）

**技術的詳細**:

**プロンプト設計**:
- Qwen3-VL cookbookの`2d_grounding.ipynb`を参照
- 推奨形式: `{"point_2d": [x, y], "gripper": "open/close"}`
- JSON形式でstructured出力を実現
- HAMSTER形式への変換ロジック実装

**VILA vs Qwen3プロンプト比較**:

VILA (HAMSTER):
```
In the image, please execute the command described in <quest>{task}</quest>.
Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.
Format your answer as a list of tuples enclosed by <ans> and </ans> tags.
...
Remember to provide points between <ans> and </ans> tags and think step by step.
```

Qwen3 (最適化版):
```
In the image, execute the task described in <quest>{task}</quest>.
Generate a sequence of waypoints representing the robot gripper's trajectory to achieve the goal.
For each waypoint, report the point coordinates and gripper action in JSON format like this:
[{"point_2d": [x, y], "gripper": "open/close"}, ...]
...
Think step by step and output the complete trajectory in JSON format.
```

**依存関係**:
- Phase 2完了（データセット） ✅
- Phase 3完了（エンコーダ） ✅

**実際の作業時間**: 約3時間

**結論**:
Qwen3-VLのゼロショット性能では**ロボット軌道生成タスクに不十分**。プロンプト最適化だけでは解決できず、VILAのような大規模ロボットデータでのファインチューニングが必須。**Phase 4ではVILA（ファインチューニング済み）を継続使用**し、ManiFlowトレーニングに進む。

#### プロンプトエンジニアリング試行（2025-11-18〜11-19）

**目的**: Qwen3-VLのゼロショット性能をプロンプト最適化で改善する試み

**試行回数**: VERSION 1-11（11バージョン）

**タスク**: "Pick up the apple and put it behind the hammer"

**ベースライン**:
- **VILA-1.5-13B**: グリッパーが実際に通る軌跡を正確に表現（物体へのアプローチ、把持、移動、配置の各動作を含む）

**Qwen3-VL試行結果**:

| VERSION | 主な変更点 | ウェイポイント数 | 軌跡の特徴 | 結果 |
|---------|-----------|----------------|-----------|------|
| VERSION 1 | 基本改善プロンプト | 4 | 同一位置でグリッパー開閉のみ | ❌ 移動なし |
| VERSION 2 | 軌道重視プロンプト | 8 | 6ユニーク位置、軌跡の多様性あり | ✅ 最良の軌跡表現 |
| VERSION 3 | 変数座標表記 | 31 | y座標固定の水平直線 | ❌ 非現実的な軌跡 |
| VERSION 5 | 簡略プロンプト | 5 | 直線的、単純な2位置間移動 | ⚠️ 軌跡が単調 |
| VERSION 6 | "unchanged"状態追加 | 3 | 直線的、2位置間移動 | ⚠️ 軌跡が単調 |
| VERSION 7 | HAMSTER形式プロンプト | 2 | 直線的、始点→終点のみ | ⚠️ 中間動作なし |
| VERSION 8 | VILAシステムプロンプト | 2 | 直線的、始点→終点のみ | ⚠️ 中間動作なし |
| VERSION 9 | ロボット特化システムプロンプト | 3 | 3位置を通る軌跡 | ⚠️ まだ単純 |
| VERSION 10 | 0~1000座標系 | 2 | 直線的、始点→終点のみ | ⚠️ 中間動作なし |
| VERSION 11 | 中間点生成指示 | 3 | 3位置を通る軌跡 | ⚠️ まだ単純 |

**主な発見**:

1. **軌跡表現の課題**:
   - ほぼ全バージョンで直線的な軌跡に留まる（始点→終点の単純移動）
   - VERSION 2のみ、物体へのアプローチや中間動作を含む軌跡を生成
   - タスク理解不足：「リンゴを拾ってハンマーの後ろに置く」という動作の分解ができていない

2. **システムプロンプトの限定的効果**:
   - ロボット特化システムプロンプト（VERSION 9）や中間点生成指示（VERSION 11）で若干の改善
   - しかし根本的な軌跡生成能力の向上にはつながらず（依然として直線的）

3. **座標系の影響**:
   - Qwen3-VL推奨の[0, 1000]座標系（VERSION 10）は効果なし
   - 座標系の変更では軌跡の質は改善されない

4. **根本的な課題**:
   - **軌跡の質的差異**: VILAは物体操作の動作フローを理解した軌跡を生成、Qwen3は単純な始点→終点移動のみ
   - **タスク理解の深さの差**: VILAの1.2Mロボットデータでの学習が、動作の分解と軌跡生成に不可欠
   - **ゼロショットの限界**: プロンプト最適化だけでは、グリッパーが実際に通るべき軌跡を生成できない

**技術的改善点**:
- システムメッセージサポートをserver_qwen3.pyに実装（VERSION 8以降）
- 座標変換ロジック実装（[0, 1000] → [0, 1] 変換）
- プロンプト履歴管理システム構築（PROMPT_HISTORY.md作成）

**現在の状況**:
- VERSION 2が軌跡の質では最良だが、それでもVILAの生成する軌跡には及ばず
- Qwen3は始点と終点の直線的な移動しか生成できず、物体操作に必要な中間動作（アプローチ、持ち上げ、配置など）の表現が不足
- ファインチューニングなしでは、グリッパーが実際に通るべき軌跡の生成は困難
- プロンプトエンジニアリングの限界を確認

**今後の方針**:
VILAの使用を継続し、Qwen3-VLについては以下のいずれかを検討：
1. ロボットタスク特化のファインチューニング（大規模データ必要）
2. Few-shotプロンプティング（成功例の提示）
3. Chain-of-Thought等の推論強化手法

**ドキュメント**:
- 詳細なプロンプト履歴: `HAMSTER/tests/PROMPT_HISTORY.md`
- タスク履歴: `TASK_HISTORY.md` (VERSION 1-11全記録)

---

### Phase 3.6: 動画でのQwen3パス生成評価 (✅ Stage 2完了: 2025-11-25)

#### 目的
RoboTwin環境の動画データを用いて、Qwen3-VLの時間的ロバスト性を評価する。各フレームに対してパス生成を行い、物体・ロボットの位置変化に対する一貫性を検証する。

#### タスク (Stage 1: 単一エピソード検証) ✅ 完了
- [x] フレーム抽出スクリプト実装
  - [x] `HAMSTER/tests/extract_episode_frames.py` (Zarrから動画フレーム抽出)
- [x] パス生成スクリプト実装
  - [x] `HAMSTER/tests/generate_paths_for_video.py` (各フレームにQwen3適用)
- [x] パス可視化スクリプト実装
  - [x] `HAMSTER/tests/visualize_paths_on_video.py` (HAMSTER標準描画)
- [x] 動画生成スクリプト実装
  - [x] `HAMSTER/tests/create_path_video.py` (MP4出力)
- [x] 単一エピソードテスト
  - [x] pick_apple_messy エピソード0 (159フレーム)
  - [x] 動画出力確認 (`qwen3_path_video.mp4`)

#### Stage 2: 全データセット拡張 ✅ 完了 (2025-11-25)
- [x] バッチ処理スクリプト実装
  - [x] `HAMSTER/tests/batch_process_episodes.py` (6タスク × 2エピソード = 12エピソード)
  - [x] 進捗保存・再開機能
  - [x] エラーハンドリング
  - [x] タスク別ディレクトリ構造: `results/video_path_test/{task_name}/episode_{idx:02d}/`
- [x] 12エピソード分の動画生成
  - [x] 各エピソードのフレーム抽出、パス生成、可視化、動画作成
  - [x] 全12エピソード成功 (dual_bottles_pick_hard/episode_00は手動修正で完了)

**Stage 2 結果**:
| タスク | episode_00 | episode_01 |
|--------|------------|------------|
| pick_apple_messy | ✅ | ✅ |
| diverse_bottles_pick | ✅ | ✅ |
| dual_bottles_pick_hard | ✅ | ✅ |
| empty_cup_place | ✅ | ✅ |
| block_hammer_beat | ✅ | ✅ |
| shoe_place | ✅ | ✅ |

#### Stage 2.5: Bimanualプロンプト拡張 ✅ 完了 (2025-11-25)
- [x] VERSION 19 (Bimanual) プロンプト設計
  - [x] `<left_arm>`/`<right_arm>`タグによる2アーム分離出力
  - [x] VERSION 18ベースで拡張
- [x] Bimanualパス生成スクリプト
  - [x] `HAMSTER/tests/generate_paths_for_video_bimanual.py`
- [x] Bimanual可視化スクリプト
  - [x] `HAMSTER/tests/visualize_paths_on_video_bimanual.py`
  - [x] HAMSTER標準描画スタイル (緑ライン、赤/青マーカー)
- [x] dual_bottles_pick_hard episode_00でテスト
  - [x] 206フレーム中204成功 (99.0%)
  - [x] 動画出力: `dual_bottles_pick_hard_bimanual/episode_00/qwen3_bimanual_path_video.mp4`

#### Stage 3: RoboTwin 2.0環境 (🔄 進行中: 2025-11-25)
- [x] RoboTwin 2.0セットアップ
  - [x] リポジトリクローン (`ManiFlow/third_party/RoboTwin2.0/`)
  - [x] アセットダウンロード (背景、ロボット、オブジェクト)
  - [x] 6タスク選択・データダウンロード (119GB)
    - beat_block_hammer, pick_diverse_bottles, pick_dual_bottles
    - place_shoe, place_empty_cup, hanging_mug
- [x] フレーム抽出スクリプト実装
  - [x] `extract_episode_frames_robotwin2.py` (HDF5→PNG)
  - [x] `batch_extract_robotwin2_frames.py` (バッチ処理)
  - [x] 12エピソード分フレーム抽出完了 (2,122フレーム)
- [x] パス生成スクリプト実装
  - [x] `generate_paths_robotwin2.py` (初期・最終フレームのみ - 完了)
  - [x] `generate_paths_robotwin2_full.py` (全フレーム対応)
- [x] 動画生成スクリプト実装
  - [x] `create_video_robotwin2.py` (パス可視化・MP4出力)
- [ ] 全フレームパス生成実行
  - [x] beat_block_hammer episode_00: 34/126フレーム (27%) 処理中断
  - [ ] 残り11エピソード
- [ ] 動画生成実行
- [ ] 評価レポート作成

**使用モデル**: Qwen3-VL-8B-Instruct
**プロンプト**:
- Single-arm: VERSION 18 (本番用)
- Bimanual: VERSION 19 (dual-arm用)

**タスク別instructions** (Stage 2で使用):
```python
TASK_INSTRUCTIONS = {
    'pick_apple_messy': 'Pick up the apple from the messy table',
    'diverse_bottles_pick': 'Pick up the bottles from the table',
    'dual_bottles_pick_hard': 'Pick up two bottles with both hands',
    'empty_cup_place': 'Place the empty cup on the target location',
    'block_hammer_beat': 'Pick up the hammer and beat the block',
    'shoe_place': 'Place the shoe on the target location',
}
```

**データセット**:
- **RoboTwin 1.0**: 6タスク (pick_apple_messy, diverse_bottles_pick, dual_bottles_pick_hard, empty_cup_place, block_hammer_beat, shoe_place)
- **エピソード数**: Stage 1 (1エピソード) → Stage 2 (12エピソード = 6タスク×2エピソード)

**処理時間実績**:
- Stage 1: ~5.3分 (159フレーム × 2秒/フレーム) ✅ 完了
- Stage 2: ~2.5時間 (12エピソード) ✅ 完了
- Bimanual: ~10分 (206フレーム × 2.9秒/フレーム) ✅ 完了

**依存関係**:
- Phase 3.5完了（Qwen3統合） ✅
- VERSION 18プロンプト確定 ✅
- RoboTwin 1.0データセット ✅

**成果物**:
- Stage 1: 1本の動画 (pick_apple_messy episode 0) ✅ 完了
- Stage 2: 12本の動画 (6タスク×2エピソード) ✅ 完了
- Stage 2.5: 1本のbimanual動画 (dual_bottles_pick_hard episode 0) ✅ 完了
- Stage 3: RoboTwin 2.0評価レポート ⏳ 未着手

**データ格納構造** (Stage 2完了後):
```
HAMSTER/results/video_path_test/
├── episode_0/                      # Stage 1 (完了)
├── batch_progress.json             # Stage 2進捗管理
├── pick_apple_messy/
│   ├── episode_00/                 # ✅
│   └── episode_01/                 # ✅
├── diverse_bottles_pick/
│   ├── episode_00/                 # ✅
│   └── episode_01/                 # ✅
├── dual_bottles_pick_hard/
│   ├── episode_00/                 # ✅
│   └── episode_01/                 # ✅
├── empty_cup_place/
│   ├── episode_00/                 # ✅
│   └── episode_01/                 # ✅
├── block_hammer_beat/
│   ├── episode_00/                 # ✅
│   └── episode_01/                 # ✅
├── shoe_place/
│   ├── episode_00/                 # ✅
│   └── episode_01/                 # ✅
└── dual_bottles_pick_hard_bimanual/  # Stage 2.5 (Bimanual)
    └── episode_00/                 # ✅ (VERSION 19)
        ├── frames/                 # 206フレーム
        ├── paths/                  # 204パス
        ├── frames_with_paths/      # 204フレーム
        └── qwen3_bimanual_path_video.mp4
```

**データ格納構造** (Stage 3 - RoboTwin 2.0):
```
HAMSTER/results/robotwin2_stage3/
├── extraction_results.json         # フレーム抽出結果
├── generation_summary.json         # パス生成サマリー (初期・最終フレームのみ版)
├── beat_block_hammer/
│   ├── episode_00/
│   │   ├── frames/                 # 126フレーム ✅
│   │   └── paths/                  # 34/126パス (27%) 🔄
│   └── episode_01/
│       └── frames/                 # 115フレーム ✅
├── pick_diverse_bottles/
│   ├── episode_00/frames/          # 122フレーム ✅
│   └── episode_01/frames/          # 118フレーム ✅
├── pick_dual_bottles/
│   ├── episode_00/frames/          # 125フレーム ✅
│   └── episode_01/frames/          # 132フレーム ✅
├── place_shoe/
│   ├── episode_00/frames/          # 180フレーム ✅
│   └── episode_01/frames/          # 180フレーム ✅
├── place_empty_cup/
│   ├── episode_00/frames/          # 180フレーム ✅
│   └── episode_01/frames/          # 169フレーム ✅
└── hanging_mug/
    ├── episode_00/frames/          # 338フレーム ✅
    └── episode_01/frames/          # 337フレーム ✅
```

**ドキュメント**:
- 実装計画: 本セクション (PROJECT_PROGRESS.md)
- RoboTwin概要: `docs/ROBOTWIN_OVERVIEW.md`
- プロンプト履歴: `HAMSTER/tests/PROMPT_HISTORY.md` (VERSION 1-19)

---

### Phase 3.7: Hyak HPC環境移行 (🔄 進行中: 2025-11-26)

#### 目的
ローカル開発環境（RTX 5090）からUW Hyak HPCクラスタ（A40/L40s GPU）への環境移行を行い、大規模トレーニングを可能にする。

#### 背景
- ローカル環境: RTX 5090 (32GB VRAM)、PyTorch nightly必須
- Hyak環境: A40 (48GB) / L40s (48GB)、PyTorch安定版使用可能
- 移行方法: Docker → DockerHub → Singularity変換

#### アーキテクチャ
```
[ローカルPC]                    [DockerHub]                 [Hyak HPC]
Dockerfile                         │                           │
    │                              │                           │
    ├─ GitHub push ──────────────► │                           │
    │                              │                           │
    │  GitHub Actions              │                           │
    │  (自動ビルド)                │                           │
    │       │                      │                           │
    │       └─ docker push ──────► naototo0103/hamster-maniflow:latest
    │                              │                           │
    │                              │      singularity pull ◄───┤
    │                              │           │               │
    │                              │           ▼               │
    │                              │      .sifファイル         │
    │                              │      (自動変換)           │
    │                              │           │               │
    │                              │           ▼               │
    │                              │      実行環境完成         │
```

#### タスク
- [x] Hyak移行計画の策定
- [x] Dockerfileの作成
  - [x] CUDA 12.1 + cuDNN 8ベースイメージ
  - [x] Python 3.10環境
  - [x] システム依存関係（Vulkan、OpenGL、MuJoCo）
- [x] requirements_docker.txtの作成
  - [x] PyTorch依存関係
  - [x] ManiFlow依存関係
  - [x] Qwen3-VL依存関係
- [x] GitHub Actionsワークフロー作成
  - [x] `.github/workflows/docker-build.yml`
  - [x] DockerHub自動push設定
  - [x] ディスク容量対策（不要ファイル削除）
- [x] GitHubリポジトリ作成・push
  - [x] リポジトリ: `https://github.com/naoto0103/uw-project.git`
  - [x] docker/, .github/をpush
- [x] DockerHubトークン設定
  - [x] アクセストークン作成
  - [x] GitHub Secretsに`DOCKERHUB_TOKEN`追加
- [x] サブディレクトリの.git削除
  - [x] HAMSTER/.git, ManiFlow/.git等を削除
  - [x] 単一リポジトリとして管理
- [ ] GitHub Actionsでビルド成功
  - [x] PyTorchインストール修正（バージョン指定削除）
  - [x] ディスク容量確保（30GB削減）
  - [x] numpy依存関係解決（>=1.24,<2.0）
  - [ ] PyTorch3Dビルド成功
- [ ] Hyakでのセットアップ
  - [ ] Singularityイメージpull
  - [ ] インスタンス起動
  - [ ] 動作確認
- [ ] データ転送
  - [ ] コードのgit clone
  - [ ] RoboTwinデータのrsync
  - [ ] モデルのHuggingFace自動ダウンロード

#### 実装ファイル
```
HAMSTER-ManiFlow-Integration/
├── .github/
│   └── workflows/
│       └── docker-build.yml        # GitHub Actionsワークフロー
├── docker/
│   ├── Dockerfile                  # Dockerイメージ定義
│   ├── requirements_docker.txt     # Python依存関係
│   ├── hyak_setup.sh              # Hyak初期セットアップスクリプト
│   ├── initialize_container.sh     # コンテナ内初期化スクリプト
│   └── README.md                   # 使い方ドキュメント
└── .gitignore                      # 大きなファイルを除外
```

#### Dockerイメージ構成
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 主要コンポーネント:
# - Python 3.10
# - PyTorch (CUDA 12.1)
# - PyTorch3D (ソースビルド)
# - MuJoCo 2.1.0
# - Vulkan/OpenGL
# - ManiFlow依存関係
# - Qwen3-VL依存関係
```

#### 解決した技術的課題
1. **PyTorchバージョン問題**: `torch==2.4.1`がCUDA 12.1で提供されていない
   - 解決: バージョン指定を削除し、自動で互換バージョンを選択
2. **GitHub Actionsディスク容量不足**: PyTorchダウンロード中に容量不足
   - 解決: ビルド前に不要ファイル（.NET, Android SDK等）を削除して30GB確保
3. **numpy依存関係競合**: numpy==1.23.5 vs numba>=1.24要件
   - 解決: `numpy>=1.24,<2.0`に変更
4. **PyTorch3D build isolation**: ビルド時にPyTorchが見つからない
   - 解決: `--no-build-isolation`オプション追加（進行中）

#### 移行対象データ
| データ | サイズ | 移行方法 |
|--------|--------|----------|
| コード | ~300MB | git clone |
| RoboTwin 1.0データ | ~7GB | rsync |
| RoboTwin 2.0データ | ~119GB | rsync |
| HAMSTERパス(.pkl) | ~50KB | git (コードと一緒) |
| Qwen3-VLモデル | ~17GB | HuggingFace自動DL |
| VILAモデル | ~51GB | rsync or HuggingFace |

#### Hyak環境情報
- **クラスタ**: UW Hyak HPC
- **GPU**: NVIDIA A40 (48GB) / L40s (48GB)
- **コンテナ**: Singularity
- **ストレージ**: `/gscratch/scrubbed/{user}/`

#### Hyakでの使用コマンド
```bash
# GPUノード取得
srun -p gpu-a40 -A {lab_account} --nodes=1 --cpus-per-task=32 \
     --mem=400G --time=168:00:00 --gpus=2 --pty /bin/bash

# Singularityイメージpull
export SINGULARITY_CACHEDIR=/gscratch/scrubbed/${USER}/singularity/cache
module load singularity
singularity pull docker://naototo0103/hamster-maniflow:latest

# インスタンス起動
singularity instance start --nv \
    --bind /gscratch/:/gscratch/:rw \
    hamster-maniflow_latest.sif hamster_train

# コンテナに入る
singularity shell instance://hamster_train
```

#### 依存関係
- Phase 3.6完了（Qwen3評価） ✅
- GitHubアカウント ✅
- DockerHubアカウント ✅
- Hyakアカウント（未確認）

#### 現在の状況
✅ **GitHub Actionsビルド成功！** (2025-11-27)

DockerHubにイメージがpush済み:
```
naototo0103/hamster-maniflow:latest
```

---

### 🚀 Hyak環境セットアップ完全ガイド

以下の手順に従えば、Hyakで環境を完全に再現できる。

---

#### Step 1: Singularityイメージのpull

```bash
# singularityディレクトリに移動
cd ~/singularity

# イメージをpull（5-15分かかる）
singularity pull docker://naototo0103/hamster-maniflow:latest

# 確認（hamster-maniflow_latest.sif ができていればOK）
ls -lh hamster-maniflow_latest.sif
```

**注意**: イメージサイズは約10GB。ストレージquotaを確認しておくこと。
```bash
quota -s  # または df -h ~
```

---

#### Step 2: GPUノードの取得

```bash
# インタラクティブGPUノード取得（A40の例）
srun -p gpu-a40 -A {lab_account} --nodes=1 --cpus-per-task=32 \
     --mem=400G --time=24:00:00 --gpus=1 --pty /bin/bash

# L40sを使う場合
srun -p gpu-l40s -A {lab_account} --nodes=1 --cpus-per-task=32 \
     --mem=400G --time=24:00:00 --gpus=1 --pty /bin/bash
```

**パラメータ説明**:
- `-p gpu-a40`: パーティション（gpu-a40 または gpu-l40s）
- `-A {lab_account}`: 研究室のアカウント名に置き換え
- `--gpus=1`: GPU数（トレーニング時は2にするとよい）
- `--time=24:00:00`: 最大実行時間

---

#### Step 3: Singularityインスタンスの起動

```bash
# モジュールロード
module load singularity

# キャッシュディレクトリ設定（オプション）
export SINGULARITY_CACHEDIR=/gscratch/scrubbed/${USER}/singularity/cache

# インスタンス起動（--nvでGPU有効化）
singularity instance start --nv \
    --bind /gscratch/:/gscratch/:rw \
    ~/singularity/hamster-maniflow_latest.sif hamster

# インスタンス確認
singularity instance list
```

**bindオプション**: `/gscratch/`をコンテナ内にマウント。データやモデルはここに置く。

---

#### Step 4: コンテナに入る

```bash
# シェルで入る
singularity shell instance://hamster

# または直接コマンド実行
singularity exec instance://hamster python --version
singularity exec instance://hamster nvidia-smi
```

---

#### Step 5: 環境確認

コンテナ内で以下を実行して、環境が正しくセットアップされていることを確認：

```bash
# Python確認
python --version  # 3.10.x

# PyTorch + CUDA確認
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# PyTorch3D確認
python -c "import pytorch3d; print(f'PyTorch3D: {pytorch3d.__version__}')"

# その他の重要ライブラリ
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import sapien; print(f'SAPIEN: {sapien.__version__}')"
```

**期待される出力**:
```
PyTorch: 2.x.x+cu121
CUDA: True
GPU: NVIDIA A40 (または L40S)
PyTorch3D: 0.7.x
transformers: 4.46.1
SAPIEN: 3.0.0b1
```

---

#### Step 6: コードのクローン

```bash
# 作業ディレクトリを作成
mkdir -p /gscratch/scrubbed/${USER}/projects
cd /gscratch/scrubbed/${USER}/projects

# GitHubからクローン
git clone https://github.com/naoto0103/uw-project.git HAMSTER-ManiFlow-Integration
cd HAMSTER-ManiFlow-Integration
```

---

#### Step 7: データの転送

**ローカルPCから**rsyncでデータを転送：

```bash
# ローカルPCで実行（Hyakにデータを送る）

# RoboTwin 1.0データ (~7GB)
rsync -avz --progress \
    ~/HAMSTER-ManiFlow-Integration/ManiFlow/data/ \
    naoto03@klone.hyak.uw.edu:/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/data/

# RoboTwin 2.0データ (~119GB) - 必要な場合のみ
rsync -avz --progress \
    ~/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/data/ \
    naoto03@klone.hyak.uw.edu:/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/data/

# HAMSTERの生成済みパス (~50KB)
rsync -avz --progress \
    ~/HAMSTER-ManiFlow-Integration/ManiFlow/data/*/hamster_paths.pkl \
    naoto03@klone.hyak.uw.edu:/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/data/
```

**Hyak上で確認**:
```bash
ls -la /gscratch/scrubbed/${USER}/projects/HAMSTER-ManiFlow-Integration/ManiFlow/data/
```

---

#### Step 8: モデルのダウンロード

HuggingFaceからモデルを直接ダウンロード（Hyak内で実行）：

```bash
# HuggingFaceキャッシュ設定
export HF_HOME=/gscratch/scrubbed/${USER}/cache/huggingface
mkdir -p $HF_HOME

# Qwen3-VL-8B-Instruct (~17GB)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen3-VL-8B-Instruct'
print(f'Downloading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
print('Done!')
"
```

**VILAモデル**（必要な場合、~51GB）:
```bash
# rsyncでローカルから転送する方が速い
rsync -avz --progress \
    ~/HAMSTER-ManiFlow-Integration/HAMSTER/Hamster_dev/ \
    naoto03@klone.hyak.uw.edu:/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/Hamster_dev/
```

---

#### Step 9: パスの設定

`~/.bashrc`または作業開始時に設定：

```bash
# プロジェクトディレクトリ
export PROJECT_DIR=/gscratch/scrubbed/${USER}/projects/HAMSTER-ManiFlow-Integration

# HuggingFaceキャッシュ
export HF_HOME=/gscratch/scrubbed/${USER}/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# MuJoCo（コンテナ内に既に設定済み）
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# CUDA
export CUDA_HOME=/usr/local/cuda
```

---

#### Step 10: 動作確認テスト

```bash
cd $PROJECT_DIR

# ManiFlowのテスト
cd ManiFlow
python -c "
from maniflow.model.vision_3d.hamster_path_encoder import HAMSTERPathDP3Encoder
import torch

encoder = HAMSTERPathDP3Encoder(
    in_channels=3,
    out_channels=128,
    state_mlp_size=[64, 64],
    path_embed_dim=128,
    path_hidden_dim=256,
    state_dim=14
)
print('HAMSTERPathDP3Encoder loaded successfully!')

# ダミーデータでテスト
batch_size = 2
point_cloud = torch.randn(batch_size, 1024, 3)
agent_pos = torch.randn(batch_size, 14)
hamster_path = torch.randn(batch_size, 50, 3)
path_mask = torch.ones(batch_size, 50)

output = encoder(point_cloud, agent_pos, hamster_path, path_mask)
print(f'Output shape: {output.shape}')
print('Test passed!')
"
```

---

#### Step 11: トレーニング実行

```bash
cd $PROJECT_DIR/ManiFlow

# 設定ファイルを指定してトレーニング開始
python scripts/train.py \
    --config-name=hamster_maniflow_pointcloud_policy_robotwin \
    task=pick_apple_messy_hamster \
    training.device=cuda:0 \
    training.batch_size=64 \
    training.num_epochs=100
```

**マルチGPU（DeepSpeed）**:
```bash
deepspeed --num_gpus=2 scripts/train.py \
    --config-name=hamster_maniflow_pointcloud_policy_robotwin \
    task=pick_apple_messy_hamster \
    training.batch_size=128
```

---

#### Step 12: セッション終了時

```bash
# コンテナを抜ける
exit

# インスタンス停止
singularity instance stop hamster

# インスタンス確認（何も表示されなければOK）
singularity instance list
```

---

#### トラブルシューティング

**問題1: GPUが認識されない**
```bash
# --nv オプションを付けてインスタンス起動しているか確認
singularity instance start --nv ...

# nvidia-smiで確認
singularity exec instance://hamster nvidia-smi
```

**問題2: ストレージ不足**
```bash
# 使用量確認
du -sh /gscratch/scrubbed/${USER}/*

# HuggingFaceキャッシュを削除
rm -rf /gscratch/scrubbed/${USER}/cache/huggingface/hub/
```

**問題3: モジュールが見つからない**
```bash
# PYTHONPATHを設定
export PYTHONPATH=$PROJECT_DIR/ManiFlow:$PYTHONPATH
```

**問題4: MuJoCoエラー**
```bash
# EGLレンダリング設定
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

---

#### クイックスタートまとめ

毎回のセッション開始時に実行：

```bash
# 1. GPUノード取得
srun -p gpu-a40 -A {lab_account} --nodes=1 --cpus-per-task=32 \
     --mem=400G --time=24:00:00 --gpus=1 --pty /bin/bash

# 2. モジュールロード＆インスタンス起動
module load singularity
singularity instance start --nv \
    --bind /gscratch/:/gscratch/:rw \
    ~/singularity/hamster-maniflow_latest.sif hamster

# 3. コンテナに入る
singularity shell instance://hamster

# 4. 環境設定
export PROJECT_DIR=/gscratch/scrubbed/${USER}/projects/HAMSTER-ManiFlow-Integration
export HF_HOME=/gscratch/scrubbed/${USER}/cache/huggingface
export PYTHONPATH=$PROJECT_DIR/ManiFlow:$PYTHONPATH
cd $PROJECT_DIR

# 5. 作業開始！
```

---

### Phase 4: ポリシー実装 (予定: Week 4-5)

#### タスク:
- [x] 設定ファイルの作成 (Phase 3で完了)
  - [x] `hamster_maniflow_pointcloud_policy_robotwin.yaml`
  - [x] `pick_apple_messy_hamster.yaml`
- [x] HAMSTER経路生成（実データ） (Phase 2で完了)
  - [x] 全6タスクの経路生成済み (281/300エピソード成功)
- [ ] トレーニングスクリプト
  - [ ] `train_eval_hamster_maniflow.sh`
- [ ] 他タスク用の設定ファイル作成
  - [ ] `diverse_bottles_pick_hamster.yaml`
  - [ ] `dual_bottles_pick_hard_hamster.yaml`
  - [ ] 他4タスク

**依存関係**:
- Phase 2完了（データセット） ✅
- Phase 3完了（エンコーダ） ✅
- **Phase 3.5完了（Qwen3統合）** ✅

**推定作業時間**: 2-3時間（大部分が既に完了）

---

### Phase 5: トレーニングとチューニング (予定: Week 5-7)

#### タスク:
- [ ] 小規模実験（10-50エピソード）
- [ ] ハイパーパラメータ探索
- [ ] アブレーション研究
  - [ ] 経路あり vs なし
  - [ ] 経路トークン次元の影響
  - [ ] Attentionヘッド数の影響

**依存関係**:
- Phase 4完了（ポリシー実装）
- HAMSTERサーバー稼働

**推定作業時間**: 20-40時間（GPU時間含む）

---

### Phase 6: スケールアップと評価 (予定: Week 7-8)

#### タスク:
- [ ] フルスケールトレーニング
- [ ] 複数タスクでの評価
- [ ] 汎化性能テスト
- [ ] 実ロボット展開の準備

**依存関係**: Phase 5完了

**推定作業時間**: 40-80時間（GPU時間含む）

---

## 📈 進捗サマリー

### 完了率

| カテゴリ | 完了 | 総数 | 進捗率 |
|---------|------|------|--------|
| **調査・設計** | 9 | 9 | **100%** ✅ |
| **Phase 1** | 5 | 5 | **100%** ✅ |
| **Phase 2** | 5 | 5 | **100%** ✅ |
| **Phase 3** | 7 | 7 | **100%** ✅ |
| **Phase 3.5** | 10 | 10 | **100%** ✅ |
| **Phase 3.6** | 11 | 14 | **79%** ✅ Stage 2完了 |
| **Phase 4** | 3 | 5 | 60% |
| **Phase 5** | 0 | 4 | 0% |
| **Phase 6** | 0 | 4 | 0% |
| **全体** | 50 | 63 | **79%** |

### タイムライン

```
Week 0 (完了) ████████████████████ 100%
  └─ 調査・設計フェーズ

Week 1 (完了) ████████████████████ 100%
  └─ Phase 1: 環境セットアップ + HAMSTER統合

Week 2 (完了) ████████████████████ 100%
  └─ Phase 2: データセット拡張 + HAMSTERパス生成

Week 3-8 (未実施) ░░░░░░░░░░░░░░░░░░░░ 0%
  ├─ Phase 3: エンコーダ実装
  ├─ Phase 4: ポリシー実装
  ├─ Phase 5: トレーニング
  └─ Phase 6: 評価
```

---

## 🗂️ プロジェクトファイル構造

### 現在のファイル構造

```
HAMSTER-ManiFlow-Integration/
├── IMPLEMENTATION_PLAN.md (59KB)     ✅ 作成済み
├── PROJECT_PROGRESS.md               ✅ 作成済み
│
├── HAMSTER/ (51GB)                   ✅ クローン済み
│   ├── VILA/                         ✅ VILAリポジトリ (132MB)
│   ├── Hamster_dev/                  ✅ VILAモデル (51GB)
│   ├── server.py                     ✅ HAMSTERサーバー (port 8000)
│   ├── gradio_server_example.py      ✅ 確認済み
│   ├── setup_server.sh               ✅ 確認済み
│   ├── test_api_client.py            ✅ Phase 1 完了
│   ├── requirements.txt              ✅ 確認済み
│   └── README.md                     ✅ 確認済み
│
└── ManiFlow/ (217MB)                 ✅ クローン済み
    ├── README.md                     ✅ 確認済み
    ├── INSTALL.md                    ✅ 確認済み
    ├── ManiFlow/maniflow/
    │   ├── policy/
    │   │   ├── base_policy.py        ✅ 確認済み
    │   │   ├── maniflow_pointcloud_policy.py  ✅ 確認済み
    │   │   └── maniflow_image_policy.py       ✅ 確認済み
    │   ├── model/
    │   │   ├── vision_3d/
    │   │   │   └── pointnet_extractor.py      ✅ 確認済み
    │   │   ├── vision_2d/
    │   │   │   └── timm_obs_encoder.py        ✅ 確認済み
    │   │   └── diffusion/
    │   │       └── ditx.py           ✅ 確認済み
    │   ├── dataset/
    │   │   ├── robotwin_dataset.py   ✅ 確認済み
    │   │   └── robotwin_image_dataset.py      ✅ 確認済み
    │   └── config/                   ✅ 確認済み
    ├── scripts/
    │   ├── train_eval_robotwin.sh    ✅ 確認済み
    │   └── ...
    └── third_party/                  ✅ 確認済み
```

### 今後作成するファイル

```
HAMSTER-ManiFlow-Integration/
│
├── HAMSTER/
│   ├── Qwen3-VL/                             ✅ Phase 3.5 完了
│   ├── server_qwen3.py                       ✅ Phase 3.5 完了 (300行)
│   ├── setup_qwen3_server.sh                 ✅ Phase 3.5 完了
│   ├── tests/                                ✅ Phase 3.5 完了
│   │   ├── test_qwen3_path.py
│   │   ├── test_qwen3_optimized.py
│   │   ├── compare_all_paths.py
│   │   ├── visualize_comparison.py
│   │   └── visualize_vila_vs_qwen3.py
│   ├── results/                              ✅ Phase 3.5 完了
│   │   ├── qwen3_test_path.pkl
│   │   ├── qwen3_optimized_path.pkl
│   │   └── visualizations/
│   └── test_api_client.py                    ✅ Phase 1 完了
│
└── ManiFlow/
    ├── tests/
    │   ├── visualize_hamster_path.py         ✅ Phase 2 完了 (可視化用)
    │   └── test_hamster_integration.py       ✅ Phase 3 完了 (432行)
    │
    ├── ManiFlow/maniflow/
    │   ├── policy/
    │   │   ├── maniflow_pointcloud_policy.py  ✅ Phase 3 修正済み (HAMSTERPathDP3Encoder対応)
    │   │   └── hamster_maniflow_policy.py     ⏳ Phase 4 (オプション)
    │   ├── model/
    │   │   └── vision_3d/
    │   │       └── hamster_path_encoder.py    ✅ Phase 3 完了 (558行)
    │   ├── dataset/
    │   │   └── hamster_robotwin_dataset.py    ✅ Phase 2 完了 (365行)
    │   └── config/
    │       ├── hamster_maniflow_pointcloud_policy_robotwin.yaml  ✅ Phase 3 完了
    │       └── robotwin_task/
    │           └── pick_apple_messy_hamster.yaml        ✅ Phase 3 完了
    ├── scripts/
    │   ├── generate_hamster_paths.py          ✅ Phase 2 完了 (503行)
    │   ├── create_synthetic_dataset.py        ✅ Phase 2 完了 (290行)
    │   ├── test_hamster_core.py               ✅ Phase 2 完了 (364行)
    │   ├── test_hamster_dataset.py            ✅ Phase 2 完了 (126行)
    │   └── train_eval_hamster_maniflow.sh     ⏳ Phase 4
    └── data/
        ├── pick_apple_messy_50.zarr           ✅ Phase 2 完了
        ├── diverse_bottles_pick_50.zarr       ✅ Phase 2 完了
        ├── dual_bottles_pick_hard_50.zarr     ✅ Phase 2 完了
        ├── empty_cup_place_50.zarr            ✅ Phase 2 完了
        ├── block_hammer_beat_50.zarr          ✅ Phase 2 完了
        ├── shoe_place_50.zarr                 ✅ Phase 2 完了
        └── {task_name}_50/hamster_paths.pkl   ✅ Phase 2 完了 (6タスク)
```

---

## 🔬 技術的知見

### HAMSTERについて（VLM高レベル経路計画）

**現在のVLM実装**:
1. **VILA-1.5-13B** (既存)
   - FastAPIサーバー（ポート8000）
   - モデルサイズ: 51GB
   - パス生成成功率: 93.7% (281/300エピソード)

2. **Qwen3-VL-8B-Instruct** (Phase 3.5で追加) ⭐新規
   - FastAPIサーバー（ポート8001）
   - モデルサイズ: ~17GB
   - RefCOCOベンチマーク: 82-87%
   - 高度な空間理解能力、2D/3Dグラウンディング

**共通アーキテクチャ**:
- 入力: RGB画像 + 自然言語
- 出力: 正規化2D座標列 + グリッパアクション

**経路フォーマット**:
```python
[(x1, y1), (x2, y2), ..., <action>Close Gripper</action>, ...]
# x, y ∈ [0, 1]
# グリッパアクション: (1000.0, 1000.0) = Close, (1001.0, 1001.0) = Open
```

**パフォーマンス**:
- 推論速度: ~5秒/画像（VILA-13B）
- エピソード開始時に1回のみ呼び出し（効率的）

---

### ManiFlowについて

**アーキテクチャ**:
- DiTX: Diffusion Transformer with Cross-Attention
- DP3Encoder: Dense Point Cloud Encoder
- Consistency Flow: 1-2ステップ推論

**入力モダリティ**:
1. **3D点群**: [B, N, 3/6] - XYZ または XYZRGB
2. **2D画像**: [B, 3, H, W] - RGB

**推論速度**:
- 3Dポリシー: ~50ms (Consistency Flow 2ステップ)
- 2Dポリシー: ~40ms

**データ効率**:
- 50エピソードで高性能を達成
- データ拡張が有効（Color Jitter、RandomCrop）

---

### 統合設計の重要ポイント

1. **経路の表現**:
   - 固定長（50点）にパディング
   - マスクで有効点を識別
   - 各点: (x, y, gripper_state)

2. **エンコーディング戦略**:
   - Token埋め込み → 位置エンコーディング → Attention集約
   - 点群特徴・状態特徴と並行処理
   - 最終的に連結

3. **データフロー**:
   ```
   HAMSTER (1回/エピソード) → 2D経路
                              ↓
   データセット → パディング → [B, To, 50, 3]
                              ↓
   PathTokenEncoder → [B, To, 256]
                              ↓
   DP3Encoder統合 → [B, To, 576]
                              ↓
   DiTX → Consistency Flow → Actions
   ```

---

## 📊 次のステップ

### 最優先タスク (Phase 1)

1. **HAMSTERサーバーのセットアップ** (推定: 4時間)
   ```bash
   # VILAクローン
   cd ~/HAMSTER-ManiFlow-Integration/HAMSTER
   git clone https://github.com/NVlabs/VILA.git
   cd VILA
   git checkout a5a380d6d09762d6f3fd0443aac6b475fba84f7e

   # 環境構築
   ./environment_setup.py vila
   conda activate vila
   pip install gradio openai opencv-python matplotlib numpy

   # サーバー起動
   cd ~/HAMSTER-ManiFlow-Integration/HAMSTER
   ./setup_server.sh
   ```

2. **接続テスト** (推定: 1時間)
   ```bash
   # utils/ディレクトリ作成
   mkdir -p ~/HAMSTER-ManiFlow-Integration/utils

   # test_hamster_connection.pyの作成と実行
   python utils/test_hamster_connection.py
   ```

3. **hamster_client.pyの実装** (推定: 2時間)
   - OpenAI API互換クライアント
   - レスポンスパーサー
   - エラーハンドリング

### 推奨作業順序

1. **Week 1**: Phase 1完了 ✅ **完了**
2. **Week 2**: Phase 2完了（データセット拡張） ✅ **完了**
3. **Week 3-4**: Phase 3（エンコーダ実装） ✅ **完了 (2025-11-16)**
4. **Week 4-5**: Phase 4（ポリシー実装） ← **次のステップ（60%完了）**
5. **Week 5-7**: Phase 5（トレーニング・チューニング）
6. **Week 7-8**: Phase 6（評価・ベンチマーク）

### Phase 3完了報告（2025-11-16）

1. **PathTokenEncoderの実装** ✅
   - トークン埋め込み層（3 → 128次元）
   - 学習可能な位置エンコーディング
   - Transformer Encoder + Query Token集約

2. **HAMSTERPathDP3Encoderの実装** ✅
   - DP3Encoder拡張（継承ではなく独立実装）
   - 特徴結合: 点群(128) + 状態(64) + 経路(256) = 448次元
   - Pointwiseモード対応

3. **ポリシー統合** ✅
   - `encoder_type="HAMSTERPathDP3Encoder"`選択肢追加
   - 設定ファイル作成完了

4. **統合テスト** ✅
   - 全5テスト成功
   - 損失計算・勾配伝播確認済み

### 次のアクション（Phase 4残タスク）

1. **トレーニングスクリプト作成**
   - `train_eval_hamster_maniflow.sh`

2. **他タスク用設定ファイル**
   - 残り5タスク分の作成

---

## 📚 参考資料

### 論文
- [HAMSTER論文 (arXiv 2502.05485)](https://arxiv.org/abs/2502.05485)
- [ManiFlow論文 (arXiv 2509.01819)](https://arxiv.org/abs/2509.01819)

### プロジェクトページ
- [HAMSTER](https://hamster-robot.github.io/)
- [ManiFlow](https://maniflow-policy.github.io/)

### GitHubリポジトリ
- [HAMSTER_beta](https://github.com/liyi14/HAMSTER_beta)
- [ManiFlow_Policy](https://github.com/geyan21/ManiFlow_Policy)
- [VILA](https://github.com/NVlabs/VILA)

### ローカルドキュメント
- [`IMPLEMENTATION_PLAN.md`](/home/naoto/HAMSTER-ManiFlow-Integration/IMPLEMENTATION_PLAN.md) - 詳細実装計画
- [`HAMSTER/README.md`](/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/README.md) - HAMSTERセットアップ
- [`ManiFlow/INSTALL.md`](/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/INSTALL.md) - ManiFlowインストール
- [`ManiFlow/README.md`](/home/naoto/HAMSTER-ManiFlow-Integration/ManiFlow/README.md) - ManiFlow使用方法
- [`docs/QWEN3_VL_MODELS.md`](/home/naoto/HAMSTER-ManiFlow-Integration/docs/QWEN3_VL_MODELS.md) - Qwen3-VLモデル一覧と比較
- [`docs/ROBOTWIN_OVERVIEW.md`](/home/naoto/HAMSTER-ManiFlow-Integration/docs/ROBOTWIN_OVERVIEW.md) - RoboTwin 1.0/2.0概要
- [`HAMSTER/tests/PROMPT_HISTORY.md`](/home/naoto/HAMSTER-ManiFlow-Integration/HAMSTER/tests/PROMPT_HISTORY.md) - Qwen3プロンプト試行履歴 (VERSION 1-18)

---

## 🔄 更新履歴

| 日付 | 内容 | 担当 |
|------|------|------|
| 2025-11-12 | プロジェクト初期セットアップ完了 | Claude Code |
| 2025-11-12 | HAMSTERリポジトリクローン | Claude Code |
| 2025-11-12 | ManiFlowリポジトリクローン | Claude Code |
| 2025-11-15 | **Phase 1完了**: RTX 5090環境でHAMSTERサーバー起動成功 | Claude Code |
| 2025-11-15 | PyTorch nightly (2.10.0.dev+cu128) + SDPAフォールバック確認 | Claude Code |
| 2025-11-15 | 2D経路生成テスト成功（2サンプル画像） | Claude Code |
| 2025-11-15 | setup_server.sh修正（自動モデルパス検出） | Claude Code |
| 2025-11-15 | **Phase 2完了**: HAMSTERRoboTwinDataset実装 (365行) | Claude Code |
| 2025-11-15 | バッチ経路生成スクリプト実装 (503行) | Claude Code |
| 2025-11-15 | RoboTwinデータセットダウンロード (6タスク, 6.93GB) | Claude Code |
| 2025-11-15 | 全6データセットへのHAMSTERパス生成 (281/300成功) | Claude Code |
| 2025-11-15 | ManiFlow環境構築 (pytorch3d 0.7.8 + PyTorch nightly) | Claude Code |
| 2025-11-12 | アーキテクチャ詳細調査完了 | Claude Code |
| 2025-11-12 | HAMSTER論文アブレーション確認 | Claude Code |
| 2025-11-12 | 実装計画書作成 | Claude Code |
| 2025-11-12 | プロジェクト進捗書作成 | Claude Code |
| 2025-11-16 | **Phase 3完了**: PathTokenEncoder + HAMSTERPathDP3Encoder実装 | Claude Code |
| 2025-11-17 | **Phase 3.5完了**: Qwen3-VL統合とゼロショット評価 | Claude Code |
| 2025-11-17 | Qwen3環境構築 (PyTorch nightly + transformers 4.57.1) | Claude Code |
| 2025-11-17 | server_qwen3.py実装 (OpenAI互換API, port 8001) | Claude Code |
| 2025-11-17 | VILA vs Qwen3性能比較評価 (ゼロショット不十分と結論) | Claude Code |
| 2025-11-17 | HAMSTERディレクトリ整理 (tests/, results/追加) | Claude Code |
| 2025-11-18〜19 | Qwen3プロンプトエンジニアリング (VERSION 1-18試行) | Claude Code |
| 2025-11-19 | VILAプロンプト完全分析 (vicuna_v1テンプレート抽出) | Claude Code |
| 2025-11-19 | Qwen3サーバーにシステムプロンプト対応追加 | Claude Code |
| 2025-11-19 | プロンプト履歴管理システム構築 (PROMPT_HISTORY.md作成) | Claude Code |
| 2025-11-19 | タスク履歴・進捗ドキュメント更新 (VERSION 1-18全記録) | Claude Code |
| 2025-11-20 | **VERSION 18を本番プロンプトとして確定** | Claude Code |
| 2025-11-20 | Qwen3-VLモデル調査・ドキュメント作成 (QWEN3_VL_MODELS.md) | Claude Code |
| 2025-11-20 | RoboTwin 1.0/2.0調査・ドキュメント作成 (ROBOTWIN_OVERVIEW.md) | Claude Code |
| 2025-11-20 | **Phase 3.6開始**: 動画でのQwen3パス生成評価計画 | Claude Code |
| 2025-11-20 | Phase 3.6実装計画策定 (Stage 1-3の3段階アプローチ) | Claude Code |
| 2025-11-25 | **Phase 3.6 Stage 2完了**: 12エピソード動画生成 | Claude Code |
| 2025-11-25 | **Phase 3.6 Stage 2.5完了**: Bimanual (VERSION 19) 動画生成 | Claude Code |
| 2025-11-25 | **Phase 3.6 Stage 3開始**: RoboTwin 2.0セットアップ・データダウンロード | Claude Code |
| 2025-11-25 | RoboTwin 2.0 6タスクデータダウンロード (119GB) | Claude Code |
| 2025-11-25 | フレーム抽出スクリプト実装 (extract_episode_frames_robotwin2.py) | Claude Code |
| 2025-11-25 | バッチフレーム抽出実行 (12エピソード、2,122フレーム) | Claude Code |
| 2025-11-25 | 全フレームパス生成スクリプト実装 (generate_paths_robotwin2_full.py) | Claude Code |
| 2025-11-25 | 動画生成スクリプト実装 (create_video_robotwin2.py) | Claude Code |
| 2025-11-25 | beat_block_hammer episode_00 パス生成中断 (34/126フレーム, 27%) | Claude Code |
| 2025-11-27 | **Phase 3.7**: GitHub Actionsビルド成功、DockerHubにイメージpush完了 | Claude Code |
| 2025-11-27 | Hyak環境セットアップ完全ガイド作成 (Step 1-12 + トラブルシューティング) | Claude Code |

---

**プロジェクトステータス**: 🟢 **Phase 3.7進行中（Hyakセットアップ待ち）、Phase 4実行中（60%）**

**次回アクション**: HyakでSingularityイメージpull (`singularity pull docker://naototo0103/hamster-maniflow:latest`)

**最終更新**: 2025-11-27
