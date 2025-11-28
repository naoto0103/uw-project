# HAMSTER-ManiFlow統合プロジェクト: 実装計画書

**作成日**: 2025-11-12
**プロジェクト**: HAMSTERの高レベル経路計画とManiFlowの低レベル制御の統合

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [アーキテクチャ設計](#2-アーキテクチャ設計)
3. [実装フェーズ](#3-実装フェーズ)
4. [詳細実装仕様](#4-詳細実装仕様)
5. [ディレクトリ構造](#5-ディレクトリ構造)
6. [データフロー](#6-データフロー)
7. [トレーニング手順](#7-トレーニング手順)
8. [評価とベンチマーク](#8-評価とベンチマーク)
9. [トラブルシューティング](#9-トラブルシューティング)

---

## 1. プロジェクト概要

### 1.1 目的

HAMSTERの階層的アーキテクチャを活用し、ManiFlowの低レベルポリシーを統合することで、以下を実現する：

- **高レベル**: HAMSTER VLMによる2D経路計画とセマンティック理解
- **低レベル**: ManiFlowによる3D点群ベースの精密制御

### 1.2 科学的根拠

HAMSTERの論文（Table 3）のアブレーション研究により、以下が実証されている：

| 手法 | 成功率 |
|------|--------|
| 画像オーバーレイ | 0.83 |
| **別次元入力（連結）** | **1.00** |

**採用方針**: 経路情報を独立した特徴トークンとして扱い、点群データと並行して処理する。

### 1.3 技術スタック

- **HAMSTER**: VILA-1.5-13B VLM (高レベル経路生成)
- **ManiFlow**: DiTX + DP3Encoder (低レベル制御)
- **入力**: 3D点群 (1024点, XYZ) + 2D経路トークン (最大50点)
- **出力**: ロボットアクション (14次元)

---

## 2. アーキテクチャ設計

### 2.1 システム全体図

```
┌─────────────────────────────────────────────────────────────┐
│                HAMSTER (High-Level VLM)                      │
│  ┌─────────────┐                                            │
│  │ RGB Image   │────┐                                       │
│  │ + Language  │    │                                       │
│  └─────────────┘    ▼                                       │
│         ┌────────────────────────┐                          │
│         │   選択可能なVLM        │                          │
│         │ ┌──────────────────┐  │                          │
│         │ │ VILA-1.5-13B     │  │  (FastAPI Server)        │
│         │ │ port: 8000       │  │  OpenAI互換API           │
│         │ └──────────────────┘  │                          │
│         │ ┌──────────────────┐  │                          │
│         │ │ Qwen3-VL-8B ⭐   │  │  (Phase 3.5追加)         │
│         │ │ port: 8001       │  │  ゼロショット評価        │
│         │ └──────────────────┘  │                          │
│         └───────────┬────────────┘                          │
│                     │                                        │
│                     ▼                                        │
│        2D Path: [(x1,y1,g1), (x2,y2,g2), ...]              │
│        - x, y ∈ [0, 1]: 正規化座標                          │
│        - g ∈ {0, 1}: グリッパ状態                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 経路を独立トークンとして渡す
                      │
┌─────────────────────▼───────────────────────────────────────┐
│            ManiFlow (Low-Level Policy)                       │
│  ┌───────────────────────────────────────┐                  │
│  │ 入力観測:                              │                  │
│  │  - point_cloud: [B, 1024, 3]          │                  │
│  │  - agent_pos: [B, 14]                 │                  │
│  │  - hamster_path: [B, 50, 3]  ← NEW!  │                  │
│  └───────────┬───────────────────────────┘                  │
│              │                                               │
│              ▼                                               │
│      ┌──────────────────┐                                   │
│      │ HAMSTERPathDP3   │  拡張エンコーダ                   │
│      │ Encoder          │                                   │
│      └──────┬───────────┘                                   │
│             │                                                │
│             ├─► PointNet (点群) ──► [B, N, 256]            │
│             ├─► StateMLP (状態) ──► [B, 64]                │
│             └─► PathTokenEncoder (経路) ──► [B, 256]       │
│                                                              │
│              ▼                                               │
│      統合特徴: [B, N, 576] or [B, 576]                     │
│              │                                               │
│              ▼                                               │
│      ┌──────────────────┐                                   │
│      │ DiTX Transformer │                                   │
│      │ (Diffusion)      │                                   │
│      └──────┬───────────┘                                   │
│             │                                                │
│             ▼                                                │
│  Consistency Flow (1-2 steps)                               │
│             │                                                │
│             ▼                                                │
│      Actions: [B, T, 14]                                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 経路トークンエンコーディングの詳細

```python
# 経路入力: [B, M, 3]
# - M: 最大経路点数（50点）
# - 3: (x座標, y座標, グリッパ状態)

経路点 → Token Embedding (128次元) → Positional Encoding
                                    ↓
                              Attention集約
                                    ↓
                          統合経路特徴 (256次元)
                                    ↓
                         点群特徴・状態特徴と結合
```

**設計の利点**:
1. ✅ 経路情報が独立したモダリティとして保持される
2. ✅ HAMSTERの実験結果（別次元入力が優位）に基づく
3. ✅ 点群の3D幾何情報が保全される
4. ✅ Attention機構により可変長経路に対応

---

## 3. 実装フェーズ

### Phase 1: 環境セットアップとHAMSTER統合 (Week 1)

**目標**: HAMSTERサーバーの動作確認と基本的なデータ収集

#### タスク:
- [x] プロジェクトディレクトリ構造の作成
- [ ] HAMSTERサーバーのセットアップ
  - VILAリポジトリのクローン (commit: `a5a380d6d09762d6f3fd0443aac6b475fba84f7e`)
  - VILA環境の構築 (`conda activate vila`)
  - モデルチェックポイントのダウンロード (HuggingFace: `yili18/Hamster_dev`)
  - サーバー起動テスト (`./setup_server.sh`)
- [ ] HAMSTER APIクライアントの実装
  - `hamster_client.py`の作成
  - サンプル画像でのテスト
  - レスポンスパーサーの実装

**成果物**:
- `hamster_client.py`: HAMSTER APIラッパー
- `test_hamster_connection.py`: 接続テストスクリプト

---

### Phase 2: データセット拡張 (Week 2-3)

**目標**: ManiFlowのデータセットに経路情報を追加

#### タスク:
- [ ] データセットクラスの実装
  - `hamster_robotwin_dataset.py`の作成
  - 経路生成とキャッシュ機構
  - `.zarr`ファイルへの経路保存
- [ ] 経路前処理パイプライン
  - 正規化座標の処理
  - パディング（最大50点に統一）
  - グリッパ状態のエンコーディング
- [ ] データ生成スクリプト
  - 既存デモに対するバッチ経路生成
  - データ品質チェック

**成果物**:
- `ManiFlow/maniflow/dataset/hamster_robotwin_dataset.py`
- `scripts/generate_hamster_paths.py`: バッチ経路生成
- 拡張`.zarr`データセット

---

### Phase 3: エンコーダ拡張 (Week 3-4)

**目標**: 経路トークンエンコーダの実装と統合

#### タスク:
- [ ] `HAMSTERPathDP3Encoder`の実装
  - 経路トークンエンコーダ
  - Attention集約機構
  - 特徴結合ロジック
- [ ] 単体テスト
  - ダミーデータでの動作確認
  - 出力形状の検証
  - グラデーション伝播の確認
- [ ] 既存DP3Encoderとの互換性確認

**成果物**:
- `ManiFlow/maniflow/model/vision_3d/hamster_path_encoder.py`
- `tests/test_hamster_encoder.py`

---

### Phase 3.5: Qwen3-VL統合とゼロショット評価 (2025-11-17〜)

**目標**: VLM部分をQwen3-VLに置き換え、2Dパス生成精度を向上

#### タスク:
- [ ] Qwen3-VL環境セットアップ
  - conda環境`qwen3`作成
  - `transformers>=4.57.0`インストール
  - Qwen3-VLリポジトリクローン
- [ ] OpenAI互換サーバー実装
  - `HAMSTER/server_qwen3.py`作成
  - FastAPI + Transformers
  - `/v1/chat/completions`エンドポイント
- [ ] 起動スクリプト作成
  - `HAMSTER/setup_qwen3_server.sh`
  - モデル自動ダウンロード
  - サーバー起動（port 8001）
- [ ] ゼロショット評価
  - pick_apple_messyタスクでパス生成
  - VILA vs Qwen3比較
  - 可視化スクリプト（`tests/visualize_hamster_path.py`使用）

**実装アーキテクチャ**:
```
HAMSTER/
├── VILA/                         # 既存: VILAリポジトリ
├── Qwen3-VL/                     # 新規: Qwen3リポジトリ
├── Hamster_dev/                  # 既存: VILA-1.5モデル (51GB)
├── Qwen3_dev/                    # 新規: Qwen3モデル (~17GB)
│   └── Qwen3-VL-8B-Instruct/    # Hugging Face自動DL
├── server.py                     # 既存: HAMSTERサーバー
├── server_qwen3.py               # 新規: Qwen3サーバー
├── setup_server.sh               # 既存: HAMSTER起動
└── setup_qwen3_server.sh         # 新規: Qwen3起動
```

**APIインターフェース統一**:
```python
# 両サーバーで同じエンドポイント
POST http://localhost:8000/v1/chat/completions  # HAMSTER (VILA)
POST http://localhost:8001/v1/chat/completions  # Qwen3

# 切り替えは generate_hamster_paths.py で
SERVER_PORT = 8001  # Qwen3使用
MODEL_NAME = "Qwen3-VL-8B-Instruct"
```

**評価指標**:
- パス生成成功率（現在: 93.7%）
- 経路点の精度（可視化による定性評価）
- グリッパー状態推論の正確性

**成果物**:
- `HAMSTER/server_qwen3.py`
- `HAMSTER/setup_qwen3_server.sh`
- ゼロショット評価レポート
- VILA vs Qwen3比較動画

---

### Phase 4: ポリシー実装 (Week 4-5)

**目標**: 統合ポリシーの実装

#### タスク:
- [ ] `HAMSTERManiFlowPolicy`の実装
  - `maniflow_pointcloud_policy.py`を継承
  - 新しいエンコーダの統合
  - `predict_action()`の拡張
- [ ] 設定ファイルの作成
  - `hamster_maniflow_pointcloud_policy.yaml`
  - `pick_apple_messy_hamster.yaml`
- [ ] トレーニングループの確認

**成果物**:
- `ManiFlow/maniflow/policy/hamster_maniflow_policy.py`
- `ManiFlow/maniflow/config/hamster_maniflow_pointcloud_policy.yaml`
- `ManiFlow/maniflow/config/robotwin_task/pick_apple_messy_hamster.yaml`

---

### Phase 5: トレーニングとチューニング (Week 5-7)

**目標**: 小規模データセットでの概念実証

#### タスク:
- [ ] 小規模実験（50エピソード）
  - トレーニングパイプラインの検証
  - ベースラインとの比較
- [ ] ハイパーパラメータ探索
  - 経路トークン次元
  - Attentionヘッド数
  - 学習率調整
- [ ] アブレーション研究
  - 経路あり vs なし
  - 異なる経路表現方法

**成果物**:
- トレーニングログ (W&B)
- ベンチマーク結果
- ハイパーパラメータ設定

---

### Phase 6: スケールアップと評価 (Week 7-8)

**目標**: フルスケールトレーニングと包括的評価

#### タスク:
- [ ] フルデータセットでのトレーニング
- [ ] 複数タスクでの評価
- [ ] 汎化性能テスト
  - 新規オブジェクト
  - 新規背景
  - カメラ視点変化
- [ ] 実ロボット展開の準備

**成果物**:
- 訓練済みモデル
- 評価レポート
- デモビデオ

---

## 4. 詳細実装仕様

### 4.1 経路トークンエンコーダ

**ファイル**: `ManiFlow/maniflow/model/vision_3d/hamster_path_encoder.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class PathTokenEncoder(nn.Module):
    """
    HAMSTER経路を特徴ベクトルにエンコードする

    入力: [B, M, 3]
        - B: バッチサイズ
        - M: 経路点数（最大50、パディング済み）
        - 3: (x, y, gripper_state)

    出力: [B, D]
        - D: 特徴次元（256）
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # トークン埋め込み層
        self.token_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 位置エンコーディング（学習可能）
        self.max_path_length = 50
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.max_path_length, hidden_dim) * 0.02
        )

        # Attention集約
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # クエリトークン（学習可能）
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 最終投影
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, path_tokens: torch.Tensor, path_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            path_tokens: [B, M, 3] 経路座標
            path_mask: [B, M] パディング位置を示すマスク (True=有効, False=パディング)

        Returns:
            path_features: [B, output_dim] 統合経路特徴
        """
        B, M, _ = path_tokens.shape

        # トークン埋め込み
        token_embeds = self.token_embedding(path_tokens)  # [B, M, hidden_dim]

        # 位置エンコーディングの追加
        token_embeds = token_embeds + self.positional_encoding[:, :M, :]

        # Attention集約
        query = self.query_token.expand(B, -1, -1)  # [B, 1, hidden_dim]

        # Attention（パディングマスクを適用）
        if path_mask is not None:
            # key_padding_mask: [B, M] (True=無視, False=使用)
            key_padding_mask = ~path_mask
        else:
            key_padding_mask = None

        aggregated, _ = self.attention_pool(
            query=query,
            key=token_embeds,
            value=token_embeds,
            key_padding_mask=key_padding_mask
        )  # [B, 1, hidden_dim]

        aggregated = aggregated.squeeze(1)  # [B, hidden_dim]

        # 最終投影
        path_features = self.final_projection(aggregated)  # [B, output_dim]

        return path_features


class HAMSTERPathDP3Encoder(nn.Module):
    """
    DP3Encoderを拡張してHAMSTER経路情報を統合

    入力観測:
        - point_cloud: [B, N, 3/6] 点群
        - agent_pos: [B, D] ロボット状態
        - hamster_path: [B, M, 3] HAMSTER経路
        - path_mask: [B, M] 経路マスク（オプション）

    出力:
        - [B, N, output_dim] (pointwise=True)
        - [B, output_dim] (pointwise=False)
    """

    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel=256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type='pointnet',
        downsample_points=False,
        # HAMSTER経路用パラメータ
        path_token_hidden_dim=128,
        path_token_output_dim=256,
        path_attention_heads=4,
    ):
        super().__init__()

        # 既存のDP3Encoderの初期化ロジックをコピー
        # （点群エンコーダ、状態MLPなど）
        from maniflow.model.vision_3d.pointnet_extractor import DP3Encoder, PointNetEncoderXYZ, PointNetEncoderXYZRGB, create_mlp
        import manifold.model.vision_3d.point_process as point_process

        self.point_cloud_key = 'point_cloud'
        self.state_key = 'agent_pos'
        self.path_key = 'hamster_path'
        self.path_mask_key = 'path_mask'

        self.n_output_channels = out_channel

        # 点群エンコーダの設定
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        self.path_shape = observation_space.get(self.path_key, (50, 3))

        self.downsample_points = downsample_points
        if self.downsample_points:
            self.point_preprocess = point_process.fps_torch
            self.num_points = pointcloud_encoder_cfg.num_points
        else:
            self.point_preprocess = nn.Identity()
            self.num_points = self.point_cloud_shape[0]

        # 点群エンコーダ
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        # 状態MLP
        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(*create_mlp(
            self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn
        ))

        # 経路トークンエンコーダ（NEW!）
        self.path_encoder = PathTokenEncoder(
            input_dim=3,
            hidden_dim=path_token_hidden_dim,
            output_dim=path_token_output_dim,
            num_heads=path_attention_heads,
        )

        # 出力次元の更新
        self.pointwise = pointcloud_encoder_cfg.get('pointwise', False)
        self.n_output_channels += output_dim  # 状態特徴
        self.n_output_channels += path_token_output_dim  # 経路特徴

        print(f"[HAMSTERPathDP3Encoder] Output dim: {self.n_output_channels}")
        print(f"[HAMSTERPathDP3Encoder] Pointwise: {self.pointwise}")
        print(f"[HAMSTERPathDP3Encoder] Path feature dim: {path_token_output_dim}")

    def forward(self, observations: Dict) -> torch.Tensor:
        """
        Args:
            observations: 観測辞書
                - point_cloud: [B, N, 3/6]
                - agent_pos: [B, D]
                - hamster_path: [B, M, 3]
                - path_mask: [B, M] (オプション)

        Returns:
            final_features: [B, N, output_dim] or [B, output_dim]
        """
        # 点群処理
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3

        if self.downsample_points and points.shape[1] > self.num_points:
            points, _ = self.point_preprocess(points, self.num_points)

        pn_feat = self.extractor(points)  # [B, N, C] or [B, C]

        # 状態処理
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # [B, 64]

        # 経路処理（NEW!）
        path_tokens = observations[self.path_key]  # [B, M, 3]
        path_mask = observations.get(self.path_mask_key, None)  # [B, M]
        path_feat = self.path_encoder(path_tokens, path_mask)  # [B, 256]

        # 特徴の統合
        if len(pn_feat.shape) == 3:
            # Pointwise: 各点に状態と経路の特徴をブロードキャスト
            B, N, C = pn_feat.shape
            state_feat = state_feat.unsqueeze(1).expand(-1, N, -1)  # [B, N, 64]
            path_feat = path_feat.unsqueeze(1).expand(-1, N, -1)    # [B, N, 256]

        # 結合
        final_feat = torch.cat([pn_feat, state_feat, path_feat], dim=-1)

        return final_feat

    def output_shape(self):
        return self.n_output_channels
```

---

### 4.2 データセット実装

**ファイル**: `ManiFlow/maniflow/dataset/hamster_robotwin_dataset.py`

```python
import numpy as np
import torch
from typing import Dict, Optional
import requests
import base64
import cv2
from io import BytesIO
from openai import OpenAI
import re
from termcolor import cprint

from maniflow.dataset.robotwin_dataset import RoboTwinDataset


class HAMSTERRoboTwinDataset(RoboTwinDataset):
    """
    HAMSTER経路情報を統合したRoboTwinデータセット
    """

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        use_pc_color=False,
        pointcloud_color_aug_cfg=None,
        # HAMSTER関連パラメータ
        hamster_server_url="http://localhost:8000",
        hamster_model_name="HAMSTER_dev",
        use_cached_paths=True,
        max_path_points=50,
        regenerate_paths=False,  # Trueの場合、既存キャッシュを無視
        **kwargs
    ):
        # 親クラスの初期化
        super().__init__(
            zarr_path=zarr_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
            task_name=task_name,
            use_pc_color=use_pc_color,
            pointcloud_color_aug_cfg=pointcloud_color_aug_cfg,
            **kwargs
        )

        self.hamster_server_url = hamster_server_url
        self.hamster_model_name = hamster_model_name
        self.use_cached_paths = use_cached_paths
        self.max_path_points = max_path_points
        self.regenerate_paths = regenerate_paths

        # 経路キャッシュ
        self.path_cache = {}

        # キャッシュファイルの読み込み
        if use_cached_paths and not regenerate_paths:
            self._load_path_cache()

        cprint(f"[HAMSTERRoboTwinDataset] Initialized with HAMSTER server: {hamster_server_url}", "green")
        cprint(f"[HAMSTERRoboTwinDataset] Max path points: {max_path_points}", "yellow")
        cprint(f"[HAMSTERRoboTwinDataset] Path cache size: {len(self.path_cache)}", "yellow")

    def _load_path_cache(self):
        """経路キャッシュをファイルから読み込む"""
        import os
        import pickle

        cache_file = self.zarr_path.replace('.zarr', '_hamster_paths.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.path_cache = pickle.load(f)
            cprint(f"[HAMSTERRoboTwinDataset] Loaded {len(self.path_cache)} cached paths from {cache_file}", "green")
        else:
            cprint(f"[HAMSTERRoboTwinDataset] No cache file found at {cache_file}", "yellow")

    def save_path_cache(self):
        """経路キャッシュをファイルに保存"""
        import os
        import pickle

        cache_file = self.zarr_path.replace('.zarr', '_hamster_paths.pkl')

        with open(cache_file, 'wb') as f:
            pickle.dump(self.path_cache, f)

        cprint(f"[HAMSTERRoboTwinDataset] Saved {len(self.path_cache)} paths to {cache_file}", "green")

    def query_hamster(self, rgb_image: np.ndarray, task_description: str) -> np.ndarray:
        """
        HAMSTERサーバーに画像とタスク記述を送信し、2D経路を取得

        Args:
            rgb_image: [H, W, 3] RGB画像 (0-255)
            task_description: タスク記述文字列

        Returns:
            path_points: [M, 3] 経路点 (x, y, gripper_state)
        """
        try:
            # 画像をbase64エンコード
            _, encoded_image_array = cv2.imencode('.jpg', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            encoded_image = base64.b64encode(encoded_image_array.tobytes()).decode('utf-8')

            # OpenAI APIフォーマットでリクエスト
            client = OpenAI(base_url=self.hamster_server_url, api_key="fake-key")

            response = client.chat.completions.create(
                model=self.hamster_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                            {"type": "text", "text":
                                f"\nIn the image, please execute the command described in <quest>{task_description}</quest>.\n"
                                "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\n"
                                "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
                                "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close Gripper</action>, ...]</ans>\n"
                                "The tuple denotes point x and y location of the end effector in the image. The action tags indicate gripper actions.\n"
                                "Coordinates should be floats between 0 and 1, representing relative positions.\n"
                                "Remember to provide points between <ans> and </ans> tags and think step by step."
                            },
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.2,
                top_p=0.9,
                extra_body={"num_beams": 1, "use_cache": True}
            )

            # レスポンスをパース
            response_text = response.choices[0].message.content[0]['text']
            path_points = self._parse_hamster_response(response_text)

            return path_points

        except Exception as e:
            cprint(f"[HAMSTERRoboTwinDataset] Error querying HAMSTER: {e}", "red")
            # エラー時はダミー経路を返す
            return np.zeros((1, 3), dtype=np.float32)

    def _parse_hamster_response(self, response_text: str) -> np.ndarray:
        """
        HAMSTERのレスポンステキストから経路点を抽出

        Args:
            response_text: HAMSTER VLMの出力テキスト

        Returns:
            path_points: [M, 3] 経路点配列
        """
        # <ans>タグ内のコンテンツを抽出
        match = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL)
        if not match:
            cprint("[HAMSTERRoboTwinDataset] No <ans> tags found in response", "red")
            return np.zeros((1, 3), dtype=np.float32)

        ans_content = match.group(1)

        # グリッパアクションを特殊値に置換
        ans_content = ans_content.replace('<action>Close Gripper</action>', '(1000.0, 1000.0)')
        ans_content = ans_content.replace('<action>Open Gripper</action>', '(1001.0, 1001.0)')

        # タプルのリストを評価
        try:
            keypoints = eval(ans_content)
        except Exception as e:
            cprint(f"[HAMSTERRoboTwinDataset] Error parsing keypoints: {e}", "red")
            return np.zeros((1, 3), dtype=np.float32)

        # 経路点を処理
        processed_points = []
        gripper_state = 0  # 0=Open, 1=Close

        for point in keypoints:
            x, y = point

            # グリッパアクションの特殊値をチェック
            if x == y == 1000.0:
                # Close Gripper
                gripper_state = 1
                if len(processed_points) > 0:
                    processed_points[-1][2] = gripper_state
                continue
            elif x == y == 1001.0:
                # Open Gripper
                gripper_state = 0
                if len(processed_points) > 0:
                    processed_points[-1][2] = gripper_state
                continue

            # 通常の経路点
            processed_points.append([x, y, gripper_state])

        if len(processed_points) == 0:
            return np.zeros((1, 3), dtype=np.float32)

        return np.array(processed_points, dtype=np.float32)

    def _pad_or_truncate_path(self, path_points: np.ndarray) -> tuple:
        """
        経路点を固定長にパディング/切り詰め

        Args:
            path_points: [M, 3] 可変長経路点

        Returns:
            padded_path: [max_path_points, 3] 固定長経路点
            path_mask: [max_path_points] マスク (1=有効, 0=パディング)
        """
        M = len(path_points)

        if M >= self.max_path_points:
            # 切り詰め
            padded_path = path_points[:self.max_path_points]
            path_mask = np.ones(self.max_path_points, dtype=np.float32)
        else:
            # パディング
            padded_path = np.zeros((self.max_path_points, 3), dtype=np.float32)
            padded_path[:M] = path_points
            path_mask = np.zeros(self.max_path_points, dtype=np.float32)
            path_mask[:M] = 1.0

        return padded_path, path_mask

    def _sample_to_data(self, sample):
        """
        サンプルをトレーニングデータに変換（経路情報を追加）

        Args:
            sample: リプレイバッファからのサンプル

        Returns:
            data: トレーニングデータ辞書
        """
        # 親クラスのデータ取得（点群、状態、アクション）
        data = super()._sample_to_data(sample)

        # エピソードIDを取得（キャッシュキーとして使用）
        # 注: RoboTwinDatasetにはepisode_idがないため、
        # サンプルのインデックスをキーとして使用
        episode_idx = sample.get('episode_idx', id(sample))

        # キャッシュチェック
        if episode_idx in self.path_cache and not self.regenerate_paths:
            path_data = self.path_cache[episode_idx]
        else:
            # HAMSTERで経路を生成
            # 最初のタイムステップの画像を使用（RGB画像が必要）
            # 注: RoboTwinDatasetは点群のみなので、
            # head_cameraデータを別途読み込む必要がある

            # ここでは簡易的に、zarr_pathからhead_cameraデータを読み込む
            # 実際の実装では、データセットに画像も含める必要がある
            try:
                from maniflow.common.replay_buffer import ReplayBuffer

                # 画像データを含むリプレイバッファを作成
                image_buffer = ReplayBuffer.copy_from_path(
                    self.zarr_path.replace('.zarr', '_image.zarr'),
                    keys=['head_camera']
                )

                rgb_image = image_buffer['head_camera'][episode_idx][0]  # 最初のフレーム

            except Exception as e:
                cprint(f"[HAMSTERRoboTwinDataset] Could not load image for episode {episode_idx}: {e}", "yellow")
                # ダミー画像を使用
                rgb_image = np.zeros((240, 320, 3), dtype=np.uint8)

            # タスク記述
            task_description = self.task_name if self.task_name else "pick and place the object"

            # HAMSTER経路を取得
            path_points = self.query_hamster(rgb_image, task_description)

            # パディング/切り詰め
            padded_path, path_mask = self._pad_or_truncate_path(path_points)

            path_data = {
                'hamster_path': padded_path,
                'path_mask': path_mask
            }

            # キャッシュに保存
            if self.use_cached_paths:
                self.path_cache[episode_idx] = path_data

        # データに追加
        data['hamster_path'] = path_data['hamster_path']
        data['path_mask'] = path_data['path_mask']

        return data

    def __del__(self):
        """デストラクタ: キャッシュを保存"""
        if self.use_cached_paths and len(self.path_cache) > 0:
            self.save_path_cache()
```

---

### 4.3 ポリシー実装

**ファイル**: `ManiFlow/maniflow/policy/hamster_maniflow_policy.py`

```python
from typing import Dict
import torch
from termcolor import cprint

from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
from maniflow.model.vision_3d.hamster_path_encoder import HAMSTERPathDP3Encoder


class HAMSTERManiFlowPolicy(ManiFlowTransformerPointcloudPolicy):
    """
    HAMSTER経路情報を統合したManiFlowポリシー

    maniflow_pointcloud_policyを継承し、経路情報を追加入力として受け取る
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon,
        n_action_steps,
        n_obs_steps,
        # HAMSTER経路用パラメータ
        use_hamster_path=True,
        path_token_hidden_dim=128,
        path_token_output_dim=256,
        path_attention_heads=4,
        # 既存のManiFlowパラメータ
        **kwargs
    ):
        self.use_hamster_path = use_hamster_path
        self.path_token_hidden_dim = path_token_hidden_dim
        self.path_token_output_dim = path_token_output_dim
        self.path_attention_heads = path_attention_heads

        # 親クラスの初期化前にencoder_typeを設定
        if use_hamster_path:
            kwargs['encoder_type'] = 'HAMSTERPathDP3Encoder'

        # 親クラスの初期化
        super().__init__(
            shape_meta=shape_meta,
            horizon=horizon,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            **kwargs
        )

        cprint(f"[HAMSTERManiFlowPolicy] Initialized with HAMSTER path integration", "green")
        cprint(f"  - use_hamster_path: {use_hamster_path}", "yellow")
        cprint(f"  - path_token_output_dim: {path_token_output_dim}", "yellow")
        cprint(f"  - path_attention_heads: {path_attention_heads}", "yellow")

    def _create_obs_encoder(self, obs_dict, **kwargs):
        """
        観測エンコーダの作成（オーバーライド）
        """
        if self.use_hamster_path:
            # HAMSTERPathDP3Encoderを使用
            encoder = HAMSTERPathDP3Encoder(
                observation_space=obs_dict,
                img_crop_shape=kwargs.get('crop_shape'),
                out_channel=kwargs.get('encoder_output_dim', 256),
                state_mlp_size=kwargs.get('state_mlp_size', (64, 64)),
                pointcloud_encoder_cfg=kwargs.get('pointcloud_encoder_cfg'),
                use_pc_color=kwargs.get('use_pc_color', False),
                pointnet_type=kwargs.get('pointnet_type', 'pointnet'),
                downsample_points=kwargs.get('downsample_points', False),
                # HAMSTER経路パラメータ
                path_token_hidden_dim=self.path_token_hidden_dim,
                path_token_output_dim=self.path_token_output_dim,
                path_attention_heads=self.path_attention_heads,
            )
        else:
            # 標準のDP3Encoderを使用
            from maniflow.model.vision_3d.pointnet_extractor import DP3Encoder
            encoder = DP3Encoder(
                observation_space=obs_dict,
                img_crop_shape=kwargs.get('crop_shape'),
                out_channel=kwargs.get('encoder_output_dim', 256),
                pointcloud_encoder_cfg=kwargs.get('pointcloud_encoder_cfg'),
                use_pc_color=kwargs.get('use_pc_color', False),
                pointnet_type=kwargs.get('pointnet_type', 'pointnet'),
                downsample_points=kwargs.get('downsample_points', False),
            )

        return encoder

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        アクション予測（経路情報を含む観測を処理）

        Args:
            obs_dict: 観測辞書
                - point_cloud: [B, To, N, 3/6]
                - agent_pos: [B, To, D]
                - hamster_path: [B, To, M, 3]  (use_hamster_path=Trueの場合)
                - path_mask: [B, To, M]  (オプション)

        Returns:
            result: アクション辞書
                - action: [B, Ta, Da]
                - action_pred: [B, T, Da]
        """
        # 親クラスのpredict_actionを呼び出し
        # （HAMSTERPathDP3Encoderが自動的に経路を処理）
        return super().predict_action(obs_dict)
```

**重要な注意**:
親クラス`ManiFlowTransformerPointcloudPolicy`の`__init__`内で`DP3Encoder`が直接インスタンス化されているため、実際にはそこを修正する必要があります。上記の`_create_obs_encoder`メソッドは概念的な例です。

実装では、以下のいずれかの方法を取ります：

**方法A**: `ManiFlowTransformerPointcloudPolicy.__init__`を修正して、`encoder_type`に基づいてエンコーダを選択できるようにする

**方法B**: `HAMSTERManiFlowPolicy.__init__`で`super().__init__`の前に`obs_encoder`を作成し、引数として渡す

---

### 4.4 設定ファイル

**ファイル**: `ManiFlow/maniflow/config/hamster_maniflow_pointcloud_policy.yaml`

```yaml
defaults:
  - robotwin_task: pick_apple_messy_hamster

name: train_hamster_maniflow_pointcloud_policy

task_name: ${robotwin_task.name}
shape_meta: ${robotwin_task.shape_meta}
exp_name: "hamster_maniflow_debug"

horizon: 16
n_obs_steps: 2
n_action_steps: 16
n_latency_steps: 0
dataset_obs_steps: ${n_obs_steps}
keypoint_visible_rate: 1.0
obs_as_global_cond: True

policy:
  _target_: maniflow.policy.hamster_maniflow_policy.HAMSTERManiFlowPolicy

  # DiTXパラメータ
  block_type: "DiTX"
  n_layer: 12
  n_head: 8
  n_emb: 768
  visual_cond_len: 128
  max_lang_cond_len: 1024
  qkv_bias: true
  qk_norm: true

  language_conditioned: false
  pre_norm_modality: false

  # Consistency Flow パラメータ
  flow_batch_ratio: 0.75
  consistency_batch_ratio: 0.25
  sample_t_mode_flow: "beta"
  sample_t_mode_consistency: "discrete"
  sample_dt_mode_consistency: "uniform"
  sample_target_t_mode: "relative"
  denoise_timesteps: 10

  # 拡散タイムステップ埋め込み
  diffusion_timestep_embed_dim: 128
  diffusion_target_t_embed_dim: 128

  # 点群エンコーダ設定
  use_point_crop: true
  crop_shape:
  - 80
  - 80
  encoder_type: "HAMSTERPathDP3Encoder"
  encoder_output_dim: 128

  horizon: ${horizon}
  n_action_steps: ${n_action_steps}
  n_obs_steps: ${n_obs_steps}
  num_inference_steps: 2  # Consistency Flow: 1-2ステップ
  obs_as_global_cond: true
  shape_meta: ${shape_meta}

  # 点群設定
  use_pc_color: false
  pointnet_type: "pointnet"
  downsample_points: true
  pointcloud_encoder_cfg:
    in_channels: 3
    out_channels: ${policy.encoder_output_dim}
    use_layernorm: true
    final_norm: layernorm
    normal_channel: false
    num_points: ${policy.visual_cond_len}
    pointwise: true

  # HAMSTER経路統合パラメータ (NEW!)
  use_hamster_path: true
  path_token_hidden_dim: 128
  path_token_output_dim: 256
  path_attention_heads: 4

# EMAモデル
ema:
  _target_: maniflow.model.diffusion.ema_model.EMAModel
  update_after_step: 0
  inv_gamma: 1.0
  power: 0.75
  min_value: 0.0
  max_value: 0.9999

# データローダー
dataloader:
  batch_size: 64  # HAMSTER経路処理のため、バッチサイズを削減
  num_workers: 8
  shuffle: True
  pin_memory: True
  persistent_workers: True

val_dataloader:
  batch_size: 64
  num_workers: 8
  shuffle: False
  pin_memory: True
  persistent_workers: False

# オプティマイザー
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 1.0e-3

# トレーニング設定
training:
  device: "cuda:0"
  env_device: "cuda:0"
  seed: 42
  debug: False
  resume: True
  lr_scheduler: cosine
  lr_warmup_steps: 500
  num_epochs: 1010
  gradient_accumulate_every: 1
  use_ema: True
  rollout_every: 500
  checkpoint_every: 100
  val_every: 50
  sample_every: 5
  max_train_steps: null
  max_val_steps: null
  tqdm_interval_sec: 1.0

# ロギング
logging:
  group: ${exp_name}
  id: null
  mode: online
  name: ${training.seed}
  project: HAMSTER_ManiFlow
  resume: true
  tags:
  - ${logging.project}
  - hamster_integration

# チェックポイント
checkpoint:
  save_ckpt: True
  topk:
    monitor_key: val_loss
    mode: min
    k: 3
    format_str: 'epoch={epoch:04d}-val_loss={val_loss:.6f}.ckpt'
  save_last_ckpt: True
  save_last_snapshot: False

# マルチラン設定
multi_run:
  run_dir: data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
  wandb_name_base: ${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}

hydra:
  job:
    override_dirname: ${name}
  run:
    dir: data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
  sweep:
    dir: data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}
    subdir: ${hydra.job.num}
```

---

**ファイル**: `ManiFlow/maniflow/config/robotwin_task/pick_apple_messy_hamster.yaml`

```yaml
defaults:
  - ../robotwin_env/pick_apple_messy@robotwin_env

name: pick_apple_messy_hamster

shape_meta: &shape_meta
  obs:
    # 点群入力
    point_cloud:
      shape: [1024, 3]  # XYZ座標のみ（色情報なし）
      type: point_cloud
    # ロボット状態
    agent_pos:
      shape: [14]
      type: low_dim
    # HAMSTER経路入力 (NEW!)
    hamster_path:
      shape: [50, 3]  # 最大50点、各点は(x, y, gripper_state)
      type: low_dim
    # 経路マスク (NEW!)
    path_mask:
      shape: [50]
      type: low_dim
  action:
    shape: [14]

env_runner:
  _target_: maniflow.env_runner.robot_runner.RobotRunner
  eval_episodes: 10
  max_steps: 300
  n_obs_steps: ${n_obs_steps}
  n_action_steps: ${n_action_steps}
  task_name: pick_apple_messy
  task_config: ${robotwin_task.robotwin_env}

dataset:
  _target_: maniflow.dataset.hamster_robotwin_dataset.HAMSTERRoboTwinDataset
  zarr_path: data/pick_apple_messy_50.zarr
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 0
  val_ratio: 0.02
  max_train_episodes: null

  # 点群設定
  use_pc_color: ${policy.use_pc_color}
  pointcloud_color_aug_cfg:
    aug_color: true
    prob: 0.2
    params: [0.3, 0.4, 0.5, 0.08]

  # HAMSTER設定 (NEW!)
  hamster_server_url: "http://localhost:8000"
  hamster_model_name: "HAMSTER_dev"
  use_cached_paths: true
  max_path_points: 50
  regenerate_paths: false
```

---

### 4.5 トレーニングスクリプト

**ファイル**: `scripts/train_eval_hamster_maniflow.sh`

```bash
#!/bin/bash

# HAMSTER-ManiFlow統合トレーニングスクリプト
# 使用例:
# bash scripts/train_eval_hamster_maniflow.sh pick_apple_messy_hamster 0901 0 0

# デバッグモード
DEBUG=False
save_ckpt=True
train=True
eval=True

# タスク設定
task_name=${1:-"pick_apple_messy_hamster"}
addition_info=${2:-"debug"}
seed=${3:-0}
gpu_id=${4:-0}

# トレーニング/評価パラメータ
eval_episode=100
eval_mode="latest"  # "best" or "latest"
num_inference_steps=2  # Consistency Flow: 1-2ステップ
n_obs_steps=2
horizon=16
n_action_steps=16

# 引数チェック
if [[ -z "$task_name" || -z "$addition_info" || -z "$seed" || -z "$gpu_id" ]]; then
    echo "Usage: $0 <task_name> <addition_info> <seed> <gpu_id>"
    echo "Example: $0 pick_apple_messy_hamster 0901 0 0"
    exit 1
fi

# タスク名処理
processed_task_name=${task_name}
if [[ $task_name == *"_hamster"* ]]; then
    processed_task_name=${task_name//_hamster/}
fi

# パス設定
base_path="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_zarr_name=${processed_task_name}
zarr_path="${base_path}/ManiFlow/data/${task_zarr_name}.zarr"
exp_name=${task_name}-hamster_maniflow-${addition_info}
run_dir="${base_path}/ManiFlow/data/outputs/${exp_name}_seed${seed}"
config_name="hamster_maniflow_pointcloud_policy"

echo "========================================"
echo "HAMSTER-ManiFlow Training Configuration"
echo "========================================"
echo "Task: ${task_name}"
echo "Experiment: ${exp_name}"
echo "Seed: ${seed}"
echo "GPU: ${gpu_id}"
echo "Config: ${config_name}"
echo "Data: ${zarr_path}"
echo "Output: ${run_dir}"
echo "========================================"

# HAMSTER サーバー起動チェック
echo "Checking HAMSTER server..."
if ! curl -s http://localhost:8000 > /dev/null; then
    echo "WARNING: HAMSTER server is not running at localhost:8000"
    echo "Please start HAMSTER server:"
    echo "  cd ~/HAMSTER-ManiFlow-Integration/HAMSTER"
    echo "  conda activate vila"
    echo "  ./setup_server.sh"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "HAMSTER server is running ✓"
fi

# データセット存在チェック
if [ ! -d "$zarr_path" ]; then
    echo "ERROR: Dataset not found at ${zarr_path}"
    echo "Please generate dataset first:"
    echo "  bash scripts/gen_demonstrations_robotwin1.0.sh ${processed_task_name} 0"
    exit 1
fi

# トレーニング実行
if [ "$train" = True ]; then
    echo ""
    echo "Starting training..."
    echo ""

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    python ManiFlow/maniflow/workspace/train_maniflow.py \
        --config-name=${config_name} \
        task_name=${task_name} \
        training.seed=${seed} \
        training.device="cuda:0" \
        training.debug=${DEBUG} \
        checkpoint.save_ckpt=${save_ckpt} \
        hydra.run.dir=${run_dir} \
        exp_name=${exp_name} \
        horizon=${horizon} \
        n_obs_steps=${n_obs_steps} \
        n_action_steps=${n_action_steps} \
        policy.num_inference_steps=${num_inference_steps}
fi

# 評価実行
if [ "$eval" = True ]; then
    echo ""
    echo "Starting evaluation..."
    echo ""

    # 最新/ベストチェックポイントを検索
    if [ "$eval_mode" = "latest" ]; then
        checkpoint_path="${run_dir}/checkpoints/latest.ckpt"
    else
        checkpoint_path=$(ls -t ${run_dir}/checkpoints/*.ckpt | head -1)
    fi

    if [ ! -f "$checkpoint_path" ]; then
        echo "ERROR: Checkpoint not found at ${checkpoint_path}"
        exit 1
    fi

    echo "Evaluating checkpoint: ${checkpoint_path}"

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    python ManiFlow/maniflow/workspace/eval_maniflow.py \
        --checkpoint=${checkpoint_path} \
        --eval_episodes=${eval_episode} \
        --output_dir=${run_dir}/eval_results \
        --device="cuda:0"
fi

echo ""
echo "========================================"
echo "Training/Evaluation completed!"
echo "Results saved to: ${run_dir}"
echo "========================================"
```

---

## 5. ディレクトリ構造

```
HAMSTER-ManiFlow-Integration/
├── IMPLEMENTATION_PLAN.md              # このファイル
│
├── HAMSTER/                             # HAMSTERサブプロジェクト
│   ├── VILA/                            # VILAリポジトリ（外部依存）
│   ├── Qwen3-VL/                        # Qwen3-VLリポジトリ（Phase 3.5追加）
│   ├── Hamster_dev/                     # VILAモデルチェックポイント (51GB)
│   ├── Qwen3_dev/                       # Qwen3モデルキャッシュ (~17GB, Phase 3.5追加)
│   ├── server.py                        # HAMSTER VLMサーバー (VILA, port 8000)
│   ├── server_qwen3.py                  # Qwen3 VLMサーバー (port 8001, Phase 3.5追加)
│   ├── setup_server.sh                  # HAMSTERサーバー起動スクリプト
│   ├── setup_qwen3_server.sh            # Qwen3サーバー起動 (Phase 3.5追加)
│   ├── test_api_client.py               # 共通APIクライアント
│   └── gradio_server_example.py         # デモインターフェース
│
├── ManiFlow/                            # ManiFlowサブプロジェクト
│   ├── ManiFlow/
│   │   └── maniflow/
│   │       ├── policy/
│   │       │   ├── base_policy.py
│   │       │   ├── maniflow_pointcloud_policy.py
│   │       │   └── hamster_maniflow_policy.py          # NEW!
│   │       │
│   │       ├── model/
│   │       │   └── vision_3d/
│   │       │       ├── pointnet_extractor.py
│   │       │       └── hamster_path_encoder.py         # NEW!
│   │       │
│   │       ├── dataset/
│   │       │   ├── robotwin_dataset.py
│   │       │   └── hamster_robotwin_dataset.py         # NEW!
│   │       │
│   │       └── config/
│   │           ├── hamster_maniflow_pointcloud_policy.yaml  # NEW!
│   │           └── robotwin_task/
│   │               └── pick_apple_messy_hamster.yaml   # NEW!
│   │
│   ├── scripts/
│   │   ├── train_eval_hamster_maniflow.sh              # NEW!
│   │   ├── generate_hamster_paths.py                   # NEW!
│   │   └── ...
│   │
│   └── data/
│       ├── pick_apple_messy_50.zarr                    # 既存データ
│       └── pick_apple_messy_50_hamster_paths.pkl       # NEW! 経路キャッシュ
│
└── utils/                                               # NEW! ユーティリティ
    ├── hamster_client.py                               # HAMSTER APIクライアント
    ├── test_hamster_connection.py                      # 接続テスト
    └── visualize_paths.py                              # 経路可視化
```

---

## 6. データフロー

### 6.1 トレーニング時のデータフロー

```
1. データローダー
   ├─ 点群データ読み込み: [B, To, N, 3]
   ├─ 状態データ読み込み: [B, To, 14]
   └─ HAMSTER経路取得:
       ├─ キャッシュチェック
       ├─ なければHAMSTERサーバーに問い合わせ
       │   ├─ RGB画像を取得
       │   ├─ HAMSTERに送信
       │   └─ 2D経路を受信: [(x,y,g), ...]
       ├─ 経路をパディング: [B, To, 50, 3]
       └─ マスク生成: [B, To, 50]

2. 観測辞書の構築
   {
     'point_cloud': [B, To, N, 3],
     'agent_pos': [B, To, 14],
     'hamster_path': [B, To, 50, 3],
     'path_mask': [B, To, 50]
   }

3. HAMSTERPathDP3Encoder
   ├─ 点群エンコーディング
   │   └─ PointNet: [B*To, N, 3] -> [B*To, N, 256]
   ├─ 状態エンコーディング
   │   └─ MLP: [B*To, 14] -> [B*To, 64]
   └─ 経路エンコーディング
       ├─ PathTokenEncoder: [B*To, 50, 3] -> [B*To, 256]
       │   ├─ Token埋め込み: [B*To, 50, 128]
       │   ├─ 位置エンコーディング追加
       │   ├─ Attention集約
       │   └─ 最終投影: [B*To, 256]
       └─ 特徴結合: [B*To, N, 576] or [B*To, 576]

4. DiTX Transformer
   ├─ 視覚条件: [B, To*N, 576] or [B, To, 576]
   ├─ ノイズ: [B, T, 14]
   └─ Consistency Flow (1-2ステップ)
       └─ アクション予測: [B, T, 14]

5. 損失計算
   ├─ Flow Matching Loss (75%)
   └─ Consistency Loss (25%)
```

### 6.2 推論時のデータフロー

```
1. 環境からの観測
   ├─ カメラから点群取得: [N, 3]
   ├─ ロボット状態取得: [14]
   └─ 初回のみHAMSTER経路生成:
       ├─ RGB画像をキャプチャ
       ├─ HAMSTERサーバーに送信
       └─ 2D経路を受信: [M, 3]

2. 観測バッファ
   ├─ n_obs_steps分の履歴を保持
   └─ 観測辞書を構築: [1, To, ...]

3. ポリシー推論
   └─ predict_action() -> [1, Ta, 14]

4. アクション実行
   ├─ n_action_steps分のアクションを取得
   └─ ロボットコントローラーに送信
```

---

## 7. トレーニング手順

### 7.1 事前準備

#### ステップ1: HAMSTERサーバーのセットアップ

```bash
# VILAリポジトリのクローン
cd ~/HAMSTER-ManiFlow-Integration/HAMSTER
git clone https://github.com/NVlabs/VILA.git
cd VILA
git checkout a5a380d6d09762d6f3fd0443aac6b475fba84f7e

# VILA環境の構築
./environment_setup.py vila
conda activate vila

# 追加パッケージのインストール
pip install gradio openai opencv-python matplotlib numpy

# サーバーの起動（別ターミナル）
cd ~/HAMSTER-ManiFlow-Integration/HAMSTER
./setup_server.sh

# サーバーテスト
curl http://localhost:8000
```

#### ステップ2: HAMSTERクライアントのテスト

```bash
cd ~/HAMSTER-ManiFlow-Integration
python utils/test_hamster_connection.py
```

**期待される出力**:
```
Connecting to HAMSTER server at http://localhost:8000...
✓ Server is running
Testing path generation...
✓ Path generated: 23 points
Sample path: [(0.45, 0.32, 0), (0.47, 0.35, 0), ...]
```

### 7.2 データセット準備

#### ステップ3: 経路データの生成

```bash
cd ~/HAMSTER-ManiFlow-Integration/ManiFlow

# 既存のデモンストレーションに対してHAMSTER経路を生成
python scripts/generate_hamster_paths.py \
    --zarr_path data/pick_apple_messy_50.zarr \
    --task_name "pick apple messy" \
    --hamster_url http://localhost:8000 \
    --output_cache data/pick_apple_messy_50_hamster_paths.pkl \
    --visualize  # 経路を可視化
```

**期待される処理時間**: 50エピソード × 約5秒/エピソード = 約4分

### 7.3 小規模実験

#### ステップ4: 概念実証トレーニング（10エピソード）

```bash
# デバッグモードで小規模トレーニング
bash scripts/train_eval_hamster_maniflow.sh \
    pick_apple_messy_hamster \
    poc_10ep \
    42 \
    0

# 設定:
# - 10エピソード
# - 10エポック
# - バッチサイズ: 8
# - 期待時間: 約30分
```

**検証項目**:
- [ ] トレーニングループが正常に動作
- [ ] 損失が減少傾向
- [ ] メモリ使用量が許容範囲内（< 16GB）
- [ ] W&Bログが正常に記録

### 7.4 フルスケールトレーニング

#### ステップ5: 50エピソードでのトレーニング

```bash
bash scripts/train_eval_hamster_maniflow.sh \
    pick_apple_messy_hamster \
    full_50ep \
    42 \
    0

# 設定:
# - 50エピソード
# - 1000エポック
# - バッチサイズ: 64
# - 期待時間: 約8-12時間（RTX 3090）
```

### 7.5 複数シードでのトレーニング

```bash
# シード0, 1, 2で並行実行（複数GPUがある場合）
for seed in 0 1 2; do
    bash scripts/train_eval_hamster_maniflow.sh \
        pick_apple_messy_hamster \
        multi_seed \
        ${seed} \
        ${seed} &
done
wait
```

---

## 8. 評価とベンチマーク

### 8.1 評価指標

1. **成功率 (Success Rate)**: タスク完了の割合
2. **平均ステップ数 (Avg. Steps)**: タスク完了までのステップ数
3. **衝突率 (Collision Rate)**: 環境との衝突頻度
4. **経路追従精度 (Path Following Accuracy)**: HAMSTER経路との一致度

### 8.2 ベースラインとの比較

| モデル | 成功率 | 平均ステップ | 推論時間 |
|--------|--------|--------------|----------|
| ManiFlow (baseline) | TBD | TBD | 50ms |
| ManiFlow + HAMSTER (overlay) | TBD | TBD | 55ms |
| **ManiFlow + HAMSTER (token)** | **TBD** | **TBD** | **60ms** |

### 8.3 アブレーション研究

```bash
# 1. 経路なし（ベースライン）
bash scripts/train_eval_robotwin.sh \
    maniflow_pointcloud_policy_robotwin \
    pick_apple_messy_pointcloud \
    baseline \
    0 \
    0

# 2. 経路あり（提案手法）
bash scripts/train_eval_hamster_maniflow.sh \
    pick_apple_messy_hamster \
    proposed \
    0 \
    0

# 3. 経路トークン次元の影響
for dim in 128 256 512; do
    # 設定ファイルでpath_token_output_dimを変更
    bash scripts/train_eval_hamster_maniflow.sh ...
done

# 4. Attentionヘッド数の影響
for heads in 1 2 4 8; do
    # 設定ファイルでpath_attention_headsを変更
    bash scripts/train_eval_hamster_maniflow.sh ...
done
```

### 8.4 汎化性能テスト

```bash
# 新規オブジェクト
python ManiFlow/maniflow/workspace/eval_maniflow.py \
    --checkpoint outputs/hamster_maniflow/latest.ckpt \
    --eval_mode novel_objects \
    --eval_episodes 50

# 新規背景
python ManiFlow/maniflow/workspace/eval_maniflow.py \
    --checkpoint outputs/hamster_maniflow/latest.ckpt \
    --eval_mode novel_backgrounds \
    --eval_episodes 50

# カメラ視点変化
python ManiFlow/maniflow/workspace/eval_maniflow.py \
    --checkpoint outputs/hamster_maniflow/latest.ckpt \
    --eval_mode novel_viewpoints \
    --eval_episodes 50
```

---

## 9. トラブルシューティング

### 9.1 よくある問題

#### 問題1: HAMSTERサーバーに接続できない

**症状**:
```
Error: Connection refused to http://localhost:8000
```

**解決策**:
```bash
# サーバーが起動しているか確認
curl http://localhost:8000

# サーバーを再起動
cd ~/HAMSTER-ManiFlow-Integration/HAMSTER
conda activate vila
./setup_server.sh
```

#### 問題2: CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory
```

**解決策**:
```yaml
# バッチサイズを削減（設定ファイル）
dataloader:
  batch_size: 32  # 64 -> 32

# またはgradient_accumulation
training:
  gradient_accumulate_every: 2
```

#### 問題3: 経路キャッシュが壊れている

**症状**:
```
Error loading path cache
```

**解決策**:
```bash
# キャッシュを削除して再生成
rm data/pick_apple_messy_50_hamster_paths.pkl

# データセットで regenerate_paths=True に設定
```

#### 問題4: 経路がすべて同じ

**症状**:
HAMSTERが異なる画像に対して同じ経路を生成する

**解決策**:
```python
# HAMSTERサーバーのtemperatureを上げる
response = client.chat.completions.create(
    ...
    temperature=0.5,  # 0.2 -> 0.5
    top_p=0.9
)
```

### 9.2 デバッグツール

#### 経路可視化

```bash
python utils/visualize_paths.py \
    --zarr_path data/pick_apple_messy_50.zarr \
    --cache_path data/pick_apple_messy_50_hamster_paths.pkl \
    --episode_idx 0 \
    --output_dir debug/visualizations
```

#### モデルのforward確認

```python
# テストスクリプト
python tests/test_hamster_encoder.py

# 期待される出力:
# ✓ PathTokenEncoder output shape: torch.Size([4, 256])
# ✓ HAMSTERPathDP3Encoder output shape: torch.Size([4, 128, 576])
# ✓ Gradient backpropagation: OK
```

---

## 付録

### A. 依存関係

```txt
# ManiFlow側
torch>=2.0.0
timm>=0.9.0
pytorch3d>=0.7.0
hydra-core>=1.3.0
wandb>=0.15.0
zarr>=2.16.0
opencv-python>=4.8.0
termcolor>=2.3.0

# HAMSTER側
openai>=1.0.0
gradio>=4.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
pillow>=10.0.0
```

### B. ハードウェア要件

**最小要件**:
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 32GB
- Storage: 100GB SSD

**推奨要件**:
- GPU: NVIDIA A100 (40GB VRAM) × 2
- RAM: 64GB
- Storage: 500GB NVMe SSD

### C. 参考リンク

- [HAMSTER論文](https://arxiv.org/abs/2502.05485)
- [HAMSTER プロジェクトページ](https://hamster-robot.github.io/)
- [ManiFlow論文](https://arxiv.org/abs/2509.01819)
- [ManiFlow GitHub](https://github.com/geyan21/ManiFlow_Policy)
- [VILA GitHub](https://github.com/NVlabs/VILA)

---

**最終更新**: 2025-11-12
**作成者**: Claude Code (Anthropic)
**プロジェクトリポジトリ**: `~/HAMSTER-ManiFlow-Integration/`
