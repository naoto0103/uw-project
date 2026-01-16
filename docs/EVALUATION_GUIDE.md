# ManiFlow + HAMSTER 評価ガイド

**最終更新**: 2025-12-26

---

## 概要

本ガイドでは、HAMSTER + ManiFlow統合システムの評価手順を説明する。6つの実験条件に対応した統一的な評価パイプラインを提供する。

### 実験条件マトリクス

|  | オリジナルManiFlow | Overlay (current) | Overlay (initial+current) |
|--|-------------------|-------------------|---------------------------|
| **学習: cluttered** | 条件1 | 条件2 | 条件3 |
| **学習: clean** | 条件4 | 条件5 | 条件6 |

**評価環境**: デフォルトは cluttered table で評価。`--eval_env clean` オプションで clean table での評価も可能。

**注**: 現在、実験的に cluttered と clean の両方の評価環境で評価を実施中。研究論文に clean 評価環境の結果を含めるかは、分析結果を確認してから決定する。

### 評価タスク（6タスク）

| タスク名 | 説明 |
|---------|------|
| click_bell | ベルをクリック |
| turn_switch | スイッチを切り替える |
| move_can_pot | 缶をポットの横に移動 |
| open_microwave | 電子レンジを開ける |
| adjust_bottle | ボトルを正しい向きで持ち上げる |
| beat_block_hammer | ハンマーでブロックを叩く |

---

## クイックスタート

### 0. GPUノードへの接続

**重要**: 評価にはA40 GPUが必要。L40S、H200などの新しいGPUではcuRoboの互換性問題が発生する。

```bash
# A40ノードに接続（例）
salloc -p gpu-a40 -A escience --nodes=1 --gpus=1 --mem=64G --time=4:00:00
```

### 1. 環境の準備

```bash
# warpキャッシュのシンボリックリンク設定（初回のみ）
rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp

# 環境のアクティベート
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
module load cuda/12.4.1 gcc/13.2.0
```

### ワンライナー評価コマンド（条件1: original mode、VILAなし、cluttered評価）

```bash
rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER && bash eval.sh --task beat_block_hammer --mode original --env cluttered --seed 42 --episodes 100
```

### ワンライナー評価コマンド（条件1: original mode、VILAなし、clean評価）

```bash
rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER && bash eval.sh --task beat_block_hammer --mode original --env cluttered --eval_env clean --seed 42 --episodes 100
```

### 2. VILAサーバーの起動（条件2,3,5,6のみ必要）

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER
bash start_vila_server.sh 0 8000
```

- GPU 0で起動、ポート8000
- 起動に約3-4分かかる
- 「Server READY!」が表示されたら準備完了
- VRAM使用量: 約26GB

### 3. 評価の実行

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER

# 条件1: cluttered + original（ベースライン、VILAなし）
bash eval.sh --task click_bell --mode original --env cluttered --seed 42

# 条件2: cluttered + overlay current
bash eval.sh --task click_bell --mode current --env cluttered --seed 42

# 条件3: cluttered + overlay initial+current（Memory Function）
bash eval.sh --task click_bell --mode initial_current --env cluttered --seed 42

# 条件4: clean学習 → cluttered評価（original）
bash eval.sh --task click_bell --mode original --env clean --seed 42

# 条件5: clean学習 → cluttered評価（current）
bash eval.sh --task click_bell --mode current --env clean --seed 42

# 条件6: clean学習 → cluttered評価（initial+current）
bash eval.sh --task click_bell --mode initial_current --env clean --seed 42

# clean環境での評価（任意の条件で --eval_env clean を追加）
bash eval.sh --task click_bell --mode original --env cluttered --eval_env clean --seed 42
```

### 4. VILAサーバーの停止

```bash
bash stop_vila_server.sh
```

---

## 評価環境

### 評価環境の選択

`--eval_env` オプションで評価環境を選択できる：

| eval_env | 説明 | 設定ファイル |
|----------|------|-------------|
| `cluttered` | 散らかったテーブル（デフォルト） | `single_arm_eval_cluttered.yml` |
| `clean` | きれいなテーブル | `single_arm_eval_clean.yml` |

**使用例:**
```bash
# cluttered環境で評価（デフォルト）
bash eval.sh --task click_bell --mode original --env cluttered --seed 42

# clean環境で評価
bash eval.sh --task click_bell --mode original --env cluttered --eval_env clean --seed 42
```

---

## 評価モード

### モード一覧

| モード | 入力 | VILAサーバー | 用途 |
|--------|------|-------------|------|
| `original` | 素のRGB画像 | 不要 | 条件1,4（ベースライン） |
| `current` | 現在フレームのオーバーレイ画像 | 必要 | 条件2,5 |
| `initial_current` | 初期+現在フレームのオーバーレイ画像 | 必要 | 条件3,6（Memory Function） |

### 入力形式

**original モード**:
```
image: [B, T, 3, 224, 224]      # 素のRGB画像
agent_pos: [B, T, 14]           # ロボット状態
```

**current モード**:
```
current_overlay: [B, T, 3, 224, 224]   # 現在フレームのオーバーレイ画像
agent_pos: [B, T, 14]                  # ロボット状態
```

**initial_current モード**:
```
initial_overlay: [B, T, 3, 224, 224]   # 初期フレーム(frame 0)のオーバーレイ画像
current_overlay: [B, T, 3, 224, 224]   # 現在フレームのオーバーレイ画像
agent_pos: [B, T, 14]                  # ロボット状態
```

---

## チェックポイント

### 命名規則

```
{env}_{task}/{mode}_seed{seed}/checkpoints/epoch=0500-*.ckpt
```

- `env`: `clean` または `cluttered`
- `task`: タスク名
- `mode`: `original`, `overlay_current`, `overlay_initial_current`
- `seed`: 学習時のシード（デフォルト: 42）

### ベースパス

```
/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/
```

### 使用するエポック

**すべての評価でEpoch 500のチェックポイントを使用**

---

## 評価フロー

### original モード

```
RGB画像 → ManiFlow → 16ステップのアクション
```

### current / initial_current モード

```
Step 0:   RGB画像 → VILA → パス生成 → オーバーレイ画像作成 → ManiFlow → 16アクション予測
Step 1-15:  アクション実行（パス固定）
Step 16:  新パス生成 → current_overlay更新 → 次の16アクション予測
... 繰り返し ...
```

**initial_current モードの特徴**:
- `initial_overlay`: エピソード開始時に作成、終了まで固定（Memory Function）
- `current_overlay`: 16ステップごとに更新

---

## パス生成失敗時の対処

### リトライ戦略

1. パス生成に失敗した場合、**最大2回までリトライ**
2. 2回リトライしても失敗した場合:
   - **直前に成功したパスがある**: そのパスを再利用
   - **直前に成功したパスがない（frame 0等）**: パス無し画像を入力として使用

### Frame 0での失敗

- パス無しの画像をそのまま入力として使用
- 以降のフレームでパス生成に成功したら、それを`initial_path`として設定

### 失敗のカウント

パス生成の失敗により誤った動作が発生し、タスクが失敗した場合は、通常通り失敗としてカウント。これにより、VLMのパス生成能力も含めたシステム全体の性能を評価する。

---

## 評価統計

### 評価プロトコル

- 各条件で**100エピソード × 1回**の評価を実行
- シード: 42（固定）
- 成功率を算出（標準偏差なし）

**注**: 当初は100エピソード × 5回（シード42-46）で平均・標準偏差を算出する予定だったが、時間的制約により1回のみの評価に変更。

### 実行コマンド

```bash
# 評価実行（cluttered環境）
bash eval.sh --task click_bell --mode initial_current --env cluttered --seed 42 --episodes 100

# 評価実行（clean環境）
bash eval.sh --task click_bell --mode initial_current --env cluttered --eval_env clean --seed 42 --episodes 100
```

---

## 出力形式

### ディレクトリ構造

出力ディレクトリは以下の命名規則に従う：

```
eval_results/
├── {task}/
│   ├── condition{N}_{train_env}_{mode}_eval{eval_env}/
│   │   └── run_seed{seed}_{run_number}/
│   │       ├── episodes.jsonl      # 各エピソードの詳細ログ
│   │       └── eval_results/       # 動画ファイル等
```

**命名規則の例:**
- `condition1_cluttered_original_evalcluttered/` - 条件1、cluttered学習、cluttered評価
- `condition1_cluttered_original_evalclean/` - 条件1、cluttered学習、clean評価
- `condition4_clean_original_evalcluttered/` - 条件4、clean学習、cluttered評価

**run_numberについて:**
- 同じ条件・seedで複数回評価を実行しても、結果は上書きされずに連番で保存される
- 例: `run_seed42_001/`, `run_seed42_002/`, `run_seed42_003/`

### episodes.jsonl スキーマ

```json
{
  "episode_id": 0,
  "task": "click_bell",
  "condition": "condition3_cluttered_overlay_initial_current_evalcluttered",
  "seed": 42,
  "success": true,
  "total_steps": 128,
  "path_stats": {
    "total_path_calls": 8,
    "path_successes": 7,
    "path_failures": 1,
    "retries": 1,
    "fallbacks_used": 1,
    "frame0_success": true
  },
  "timing": {
    "vila_inference_ms": [245, 238, 251],
    "maniflow_inference_ms": [12, 11, 13],
    "total_episode_ms": 15234
  },
  "failure_reason": null
}
```

**注**: 当初予定していたsummary.jsonとaggregated.jsonは、評価プロトコルの変更（100エピソード×1回）に伴い、生成しない。分析はepisodes.jsonlから直接行う。

---

## ファイル構成

### 評価システム

```
ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER/
├── eval.sh                       # 統一評価スクリプト
├── deploy_policy.py              # 統一評価インターフェース（3モード対応）
├── deploy_policy.yml             # 評価設定
├── start_vila_server.sh          # VILAサーバー起動
├── stop_vila_server.sh           # VILAサーバー停止
├── ManiFlow/
│   └── maniflow_policy.py        # ManiFlowラッパー（3モード対応）
├── hamster/
│   ├── vila_client.py            # VILA APIクライアント
│   ├── overlay_utils.py          # パス描画ユーティリティ
│   └── path_manager.py           # パスリトライ・フォールバック管理
└── utils/
    └── metrics_logger.py         # JSONL構造化ログ出力
```

### タスク設定

```
ManiFlow/third_party/RoboTwin2.0/task_config/
├── single_arm_eval_cluttered.yml  # cluttered環境評価用設定
└── single_arm_eval_clean.yml      # clean環境評価用設定
```

---

## コマンドラインオプション

```bash
bash eval.sh [OPTIONS]

必須オプション:
  --task TASK           タスク名（click_bell, turn_switch, move_can_pot,
                        open_microwave, adjust_bottle, beat_block_hammer）
  --mode MODE           評価モード（original, current, initial_current）
  --env ENV             学習データ環境（clean, cluttered）

オプション:
  --eval_env ENV        評価環境（clean, cluttered）（デフォルト: cluttered）
  --seed SEED           評価シード（デフォルト: 42）
  --episodes N          評価エピソード数（デフォルト: 100）
  --checkpoint PATH     チェックポイントパス（デフォルト: 自動検索）
  --vila_port PORT      VILAサーバーポート（デフォルト: 8000）
  --output_dir DIR      出力ディレクトリ（デフォルト: ./eval_results）
```

---

## トラブルシューティング

### よくある警告（無視して問題なし）

```
UserWarning: Failed to find Vulkan ICD file
missing pytorch3d
UserWarning: The detected CUDA version...
```

### VILAサーバーが起動しない

1. ポートが使用中でないか確認:
   ```bash
   curl -s http://localhost:8000/
   ```

2. ログを確認:
   ```bash
   tail -f logs/vila_server_gpu0_port8000.log
   ```

### パス生成が頻繁に失敗する

VILAサーバーのログを確認し、モデルの出力形式が正しいか確認する。フォールバック戦略により評価は継続されるが、頻発する場合はパス生成成功率が低下し、結果に影響する。

---

## 分析

### 分析ディレクトリ

評価結果の分析は、プロジェクトルートの `analysis/` ディレクトリでPythonスクリプトを用いて行う：

```
/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/
├── analysis/
│   ├── raw_data/                     # 評価結果の生データ
│   │   ├── click_bell/
│   │   ├── move_can_pot/
│   │   └── beat_block_hammer/
│   ├── scripts/                      # 分析用Pythonスクリプト
│   │   ├── load_results.py           # データ読み込みユーティリティ
│   │   └── utils.py                  # 共通ユーティリティ
│   ├── analyze_task.py               # 単一タスクの6条件比較分析
│   ├── analyze_all_tasks.py          # 全タスク横断サマリー
│   ├── generate_tables.py            # 論文用テーブル生成
│   ├── generate_figures.py           # 論文用図表生成
│   ├── config.py                     # パス設定、条件定義
│   └── outputs/                      # 出力ディレクトリ
│       ├── tables/                   # 生成されたテーブル
│       ├── figures/                  # 生成された図表
│       └── reports/                  # 分析レポート
```

**注**: Jupyter Notebookは使用せず、Pythonスクリプトで分析を行う（Hyak HPC環境でのVSCode SSH接続ではnotebookが実行できないため）。

### 分析対象データ

評価結果の生データは `analysis/raw_data/` に配置：

```
analysis/raw_data/
└── {task}/
    └── condition{N}_{train_env}_{mode}_eval{eval_env}/
        └── run_seed{seed}_{run_number}/
            ├── episodes.jsonl        # 100エピソードの詳細ログ
            └── eval_results/         # 動画ファイル等
```

---

## 関連ドキュメント

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - トレーニング手順
- [RESEARCH_OVERVIEW.md](research/RESEARCH_OVERVIEW.md) - 研究概要
- [NEXT_PHASE_REQUIREMENTS.md](NEXT_PHASE_REQUIREMENTS.md) - 詳細な実装計画
