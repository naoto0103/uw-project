# Clean環境評価コマンド（VILAパスモデル: C1-C6）

**作成日**: 2025-01-25

## 概要

VILAパスで学習したモデル（C1-C6）を**clean環境**で評価するコマンド一覧。
対象タスク: `beat_block_hammer`

**目的**: cluttered環境での評価結果と比較し、モデルの汎化性能を分析

---

## 前提条件

1. **A40 GPUノードに接続**
   ```bash
   salloc -p gpu-a40 -A escience --nodes=1 --gpus=1 --mem=64G --time=8:00:00
   ```

2. **VILAサーバーの起動**（current / initial_current モードで必要）
   ```bash
   cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER
   bash start_vila_server.sh 0 8000
   ```

---

## 出力ディレクトリ

結果は `eval_results_clean/` に保存（cluttered評価の `eval_results/` と区別）

---

## チェックポイントベースパス

```
/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/
```

---

## beat_block_hammer

### C1: cluttered学習 + original（VILAなし）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode original --env cluttered --eval_env clean --seed 42 --episodes 100 --output_dir ./eval_results_clean --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/cluttered_beat_block_hammer/original_seed42/checkpoints/epoch=0500-*.ckpt' > eval_clean_beat_block_hammer_C1.log 2>&1 &
```

### C2: cluttered学習 + current（VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode current --env cluttered --eval_env clean --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_clean --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/cluttered_beat_block_hammer/overlay_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_clean_beat_block_hammer_C2.log 2>&1 &
```

### C3: cluttered学習 + initial_current（Memory Function、VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode initial_current --env cluttered --eval_env clean --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_clean --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/cluttered_beat_block_hammer/overlay_initial_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_clean_beat_block_hammer_C3.log 2>&1 &
```

### C4: clean学習 + original（VILAなし）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode original --env clean --eval_env clean --seed 42 --episodes 100 --output_dir ./eval_results_clean --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/clean_beat_block_hammer/original_seed42/checkpoints/epoch=0500-*.ckpt' > eval_clean_beat_block_hammer_C4.log 2>&1 &
```

### C5: clean学習 + current（VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode current --env clean --eval_env clean --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_clean --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/clean_beat_block_hammer/overlay_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_clean_beat_block_hammer_C5.log 2>&1 &
```

### C6: clean学習 + initial_current（Memory Function、VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode initial_current --env clean --eval_env clean --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_clean --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/clean_beat_block_hammer/overlay_initial_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_clean_beat_block_hammer_C6.log 2>&1 &
```

---

## 条件マトリクス

| 条件 | 学習環境 | 学習パス | モード | 評価環境 | VILAサーバー |
|------|---------|---------|--------|---------|-------------|
| C1 | cluttered | VILA | original | **clean** | 不要 |
| C2 | cluttered | VILA | current | **clean** | 必要 |
| C3 | cluttered | VILA | initial_current | **clean** | 必要 |
| C4 | clean | VILA | original | **clean** | 不要 |
| C5 | clean | VILA | current | **clean** | 必要 |
| C6 | clean | VILA | initial_current | **clean** | 必要 |

---

## 実行順序の推奨

VILAサーバーの起動/停止を最小限にするため：

1. **VILAなしで実行**（C1, C4）
2. **VILAサーバー起動**
3. **VILAありで実行**（C2, C3, C5, C6）
4. **VILAサーバー停止**

---

## 注意事項

1. **チェックポイントパス**: 全コマンドで `epoch=0500-*.ckpt` のワイルドカードパターンを使用

2. **出力先**: `eval_results_clean/` に保存（既存の `eval_results/` と混同しない）

3. **ログファイル**: 各コマンドの実行ログは `eval_clean_beat_block_hammer_C{N}.log` に保存

4. **結果ディレクトリ構造**:
   ```
   eval_results_clean/
   └── beat_block_hammer/
       ├── condition1_cluttered_original_evalclean/
       ├── condition2_cluttered_overlay_current_evalclean/
       ├── condition3_cluttered_overlay_initial_current_evalclean/
       ├── condition4_clean_original_evalclean/
       ├── condition5_clean_overlay_current_evalclean/
       └── condition6_clean_overlay_initial_current_evalclean/
   ```

---

## 比較分析

評価完了後、以下の比較が可能：

| 比較 | 目的 |
|-----|-----|
| C1-C6 (eval_clean) vs C1-C6 (eval_cluttered) | 評価環境の影響 |
| C4 vs C1 (eval_clean) | In-domain vs Cross-domain (clean評価) |
| C5 vs C4, C6 vs C4 (eval_clean) | パスガイダンスの効果 (clean評価, in-domain) |
| C2 vs C1, C3 vs C1 (eval_clean) | パスガイダンスの効果 (clean評価, cross-domain) |
