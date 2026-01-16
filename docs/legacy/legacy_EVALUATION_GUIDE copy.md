# ManiFlow + HAMSTER 評価ガイド

**最終更新**: 2025-12-17

## クイックスタート

### 1. 環境の準備

```bash
# Conda環境を有効化
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
module load cuda/12.4.1 gcc/13.2.0
```

### 2. VILAサーバーの起動（ターミナル1）

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER
bash start_vila_server.sh 0 8000
```

- GPU 0で起動、ポート8000
- 起動に約3-4分かかる
- 「Server READY!」が表示されたら準備完了
- VRAM使用量: 約26GB

### 3. 評価の実行（ターミナル2）

```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
module load cuda/12.4.1 gcc/13.2.0

cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER
bash eval.sh click_bell
```

### 4. VILAサーバーの停止

```bash
bash stop_vila_server.sh
# または: kill $(cat server_pid.txt)
```

---

## ディレクトリ構成

### プロジェクト全体

```
/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/
├── HAMSTER/                    # HAMSTERオリジナルコード
│   ├── server.py               # VILAサーバー
│   └── Hamster_dev/            # 訓練済みVILAモデル
├── ManiFlow/
│   ├── ManiFlow/               # ManiFlowオリジナルコード
│   │   └── data/outputs/       # 訓練済みチェックポイント
│   └── third_party/
│       ├── RoboTwin2.0/        # RoboTwin 2.0シミュレータ
│       │   └── policy/ManiFlow_HAMSTER/  # 評価アダプター
│       └── r3m/                # R3Mエンコーダー
├── docs/                       # ドキュメント
└── singularity/                # Singularityイメージ
```

### 評価アダプター

```
ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER/
├── __init__.py               # エクスポート
├── deploy_policy.py          # RoboTwin 2.0評価インターフェース
├── deploy_policy.yml         # 評価設定
├── start_vila_server.sh      # VILAサーバー起動
├── stop_vila_server.sh       # VILAサーバー停止
├── eval.sh                   # 評価実行スクリプト
├── server_pid.txt            # サーバーPID保存
├── logs/                     # ログ出力先
├── ManiFlow/
│   ├── __init__.py
│   └── maniflow_policy.py    # ManiFlowラッパー
└── hamster/
    ├── __init__.py
    ├── vila_client.py        # VILA APIクライアント
    └── overlay_utils.py      # パス描画ユーティリティ
```

---

## Conda環境

### 環境情報

| 項目 | 値 |
|-----|-----|
| 環境名 | `robotwin` |
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4.1 |
| 場所 | `/gscratch/scrubbed/naoto03/miniconda3/envs/robotwin` |

### 主要パッケージ

- **シミュレーション**: sapien==3.0.0b1, mplib==0.2.1
- **ロボット制御**: curobo (ソースビルド)
- **ManiFlow依存**: hydra-core, dill, omegaconf, einops, timm, zarr
- **VILA依存**: openai
- **その他**: opencv-python, h5py, open3d, gymnasium

### 環境の有効化

```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
module load cuda/12.4.1 gcc/13.2.0
```

---

## 訓練済みモデル

### ManiFlowチェックポイント

| タスク | チェックポイント | エポック |
|-------|----------------|----------|
| beat_block_hammer | `2025.12.14/19.24.23_.../checkpoints/latest.ckpt` | 250 |
| click_bell | `2025.12.14/19.42.53_.../checkpoints/latest.ckpt` | 200 |
| move_can_pot | `2025.12.14/22.38.04_.../checkpoints/latest.ckpt` | 400 |

**ベースパス**: `/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/`

### VILAモデル

- **パス**: `HAMSTER/Hamster_dev/VILA1.5-13b-robopoint_1432k+rlbench_all_tasks_256_1000_eps_sketch_v5_alpha+droid_train99_sketch_v5_alpha_fix+bridge_data_v2_train90_10k_sketch_v5_alpha-e1-LR1e-5`
- **サイズ**: VILA-1.5 13B
- **用途**: 2Dパス予測

### R3Mモデル

- **パス**: `/gscratch/scrubbed/naoto03/.r3m/r3m_18`
- **ソース**: `/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/r3m` (pip install -e)

---

## モデル設定

### ManiFlowパラメータ

```yaml
horizon: 16          # 16ステップ先までのアクションを予測
n_obs_steps: 2       # 過去2フレームの観測を使用
n_action_steps: 16   # 16ステップ分のアクションを出力
```

### アーキテクチャ

- **DiTX**: 8層, 8ヘッド, 512次元, MLPratio 4.0
- **ObsEncoder**: TimmObsEncoder (R3M-ResNet18ベース)
- **総パラメータ数**: 77.93M

---

## 評価フロー

### 基本フロー

```
RGB画像 → VILA-13B → パス予測 → オーバーレイ画像 → ManiFlow → 16ステップのアクション
```

### パス生成タイミング

```
Step 0:   パス生成 → initial_overlay作成 → 16アクション予測
Step 1-15:  アクション実行（パス固定）
Step 16:  新パス生成 → current_overlay更新 → 次の16アクション予測
Step 17-31: アクション実行
... 繰り返し ...
```

**重要**:
- `initial_overlay`: エピソード開始時に作成、終了まで固定
- `current_overlay`: 16ステップごとに更新
- パス生成失敗時は前回のパスを再利用

---

## 評価結果（2025-12-17）

### 統合テスト

| タスク | 成功率 | 備考 |
|-------|--------|------|
| click_bell | 5/5 = 100% | 実行中 (100エピソード) |

### システム状態

| コンポーネント | 状態 |
|--------------|------|
| VILAサーバー | ✅ 稼働中 (GPU 0, Port 8000) |
| ManiFlowモデル | ✅ ロード成功 |
| RoboTwin 2.0 | ✅ シミュレーション正常 |
| パス生成 | ✅ 動作（一部失敗も再利用で継続） |

---

## トラブルシューティング

### よくある警告（無視して問題なし）

```
UserWarning: Failed to find Vulkan ICD file
missing pytorch3d
UserWarning: The detected CUDA version...
```

### VILAサーバーが起動しない

1. Singularityモジュールが読み込まれているか確認:
   ```bash
   module load singularity
   ```

2. ポートが使用中でないか確認:
   ```bash
   curl -s http://localhost:8000/
   ```

3. ログを確認:
   ```bash
   tail -f logs/vila_server_gpu0_port8000.log
   ```

### ModuleNotFoundError

不足パッケージをインストール:
```bash
conda activate robotwin
pip install <package_name>
```

よく不足するパッケージ: `einops`, `zarr`, `termcolor`, `timm`

### パス生成失敗

`[HAMSTERObsEncoder] WARNING: Path generation failed` が出ても、前回のパスを再利用して継続する。頻発する場合はVILAサーバーのログを確認。

---

## 利用可能なタスク

| タスク名 | 説明 | チェックポイント |
|---------|------|----------------|
| beat_block_hammer | ハンマーでブロックを叩く | ✅ |
| click_bell | ベルをクリック | ✅ |
| move_can_pot | 缶をポットの横に移動 | ✅ |
| place_object_stand | オブジェクトをスタンドに置く | ❌ |
| open_microwave | 電子レンジを開ける | ❌ |
| turn_switch | スイッチを切り替える | ❌ |

❌ = チェックポイント未訓練

---

## カスタム評価

### 特定のチェックポイントを使用

```bash
bash eval.sh click_bell --checkpoint_path /path/to/custom.ckpt
```

### 異なるシードで実行

```bash
bash eval.sh click_bell --seed 42
```

### 評価設定のカスタマイズ

`deploy_policy.yml` を編集:
```yaml
vila_server_url: "http://localhost:8000/v1"
vila_model: "HAMSTER_dev"
instruction_type: unseen
```

---

## ファイル参照

### スクリプト

| ファイル | 説明 |
|---------|------|
| `start_vila_server.sh` | VILAサーバー起動 |
| `stop_vila_server.sh` | VILAサーバー停止 |
| `eval.sh` | 評価実行 |

### Python モジュール

| ファイル | 説明 |
|---------|------|
| `deploy_policy.py` | RoboTwin評価インターフェース |
| `ManiFlow/maniflow_policy.py` | ManiFlowラッパー |
| `hamster/vila_client.py` | VILA APIクライアント |
| `hamster/overlay_utils.py` | パス描画 |

### 設定ファイル

| ファイル | 説明 |
|---------|------|
| `deploy_policy.yml` | 評価パラメータ |
| `task_config/single_arm_eval.yml` | タスク設定 |

---

## 完了項目

- [x] ManiFlow_HAMSTERアダプター作成
- [x] VILAサーバー起動/停止スクリプト
- [x] 評価設定ファイル
- [x] 評価スクリプト
- [x] 単体テスト（全コンポーネント）
- [x] **RoboTwin 2.0環境セットアップ（conda環境）**
- [x] **統合テスト実行成功**

---

## 追加情報

### Singularityイメージ

- **パス**: `/gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif`
- **サイズ**: 約12GB
- **用途**: VILAサーバー実行

### データ生成

RoboTwin 2.0でのデータ生成については `ROBOTWIN2_HYAK_DATA_GENERATION.md` を参照。

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0
python script/collect_data.py <task_name> <config_name>
```
