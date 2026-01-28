# GT Path モデル評価コマンド

**作成日**: 2025-01-20

## 概要

GT Path（Ground Truth Path）でトレーニングしたManiFlowモデルの評価コマンド一覧。
12個の条件（3タスク × 2環境 × 2手法）に対応。

**重要**: 以前のVILAパス評価結果と区別するため、`--output_dir` で `eval_results_gt` ディレクトリに出力する。

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
   - 起動に約3-4分かかる
   - 「Server READY!」が表示されたら準備完了

---

## 出力ディレクトリ

| 評価種別 | 出力ディレクトリ |
|---------|-----------------|
| VILAパス評価（以前） | `eval_results/` |
| **GTパス評価（今回）** | **`eval_results_gt/`** |

---

## チェックポイントベースパス

```
/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/
```

---

## beat_block_hammer

### GT条件2相当: cluttered学習 + current（VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode current --env cluttered --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_cluttered_beat_block_hammer/overlay_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_beat_block_hammer_cluttered_current.log 2>&1 &
```

### GT条件3相当: cluttered学習 + initial_current（Memory Function、VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode initial_current --env cluttered --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_cluttered_beat_block_hammer/overlay_initial_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_beat_block_hammer_cluttered_initial_current.log 2>&1 &
```

### GT条件5相当: clean学習 + current（VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode current --env clean --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_clean_beat_block_hammer/overlay_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_beat_block_hammer_clean_current.log 2>&1 &
```

### GT条件6相当: clean学習 + initial_current（Memory Function、VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode initial_current --env clean --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_clean_beat_block_hammer/overlay_initial_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_beat_block_hammer_clean_initial_current.log 2>&1 &
```

---

## click_bell

### GT条件2相当: cluttered学習 + current（VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task click_bell --mode current --env cluttered --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_cluttered_click_bell/overlay_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_click_bell_cluttered_current.log 2>&1 &
```

### GT条件3相当: cluttered学習 + initial_current（Memory Function、VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task click_bell --mode initial_current --env cluttered --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_cluttered_click_bell/overlay_initial_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_click_bell_cluttered_initial_current.log 2>&1 &
```

### GT条件5相当: clean学習 + current（VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task click_bell --mode current --env clean --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_clean_click_bell/overlay_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_click_bell_clean_current.log 2>&1 &
```

### GT条件6相当: clean学習 + initial_current（Memory Function、VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task click_bell --mode initial_current --env clean --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_clean_click_bell/overlay_initial_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_click_bell_clean_initial_current.log 2>&1 &
```

---

## move_can_pot

### GT条件2相当: cluttered学習 + current（VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task move_can_pot --mode current --env cluttered --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_cluttered_move_can_pot/overlay_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_move_can_pot_cluttered_current.log 2>&1 &
```

### GT条件3相当: cluttered学習 + initial_current（Memory Function、VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task move_can_pot --mode initial_current --env cluttered --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_cluttered_move_can_pot/overlay_initial_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_move_can_pot_cluttered_initial_current.log 2>&1 &
```

### GT条件5相当: clean学習 + current（VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task move_can_pot --mode current --env clean --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_clean_move_can_pot/overlay_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_move_can_pot_clean_current.log 2>&1 &
```

### GT条件6相当: clean学習 + initial_current（Memory Function、VILAあり）

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task move_can_pot --mode initial_current --env clean --eval_env cluttered --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_clean_move_can_pot/overlay_initial_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_move_can_pot_clean_initial_current.log 2>&1 &
```

---

## 条件マトリクス

| タスク | 環境 | モード | GT条件 | 対応するVILAパス条件 |
|--------|------|--------|--------|---------------------|
| beat_block_hammer | cluttered | current | GT-C2 | C2 |
| beat_block_hammer | cluttered | initial_current | GT-C3 | C3 |
| beat_block_hammer | clean | current | GT-C5 | C5 |
| beat_block_hammer | clean | initial_current | GT-C6 | C6 |
| click_bell | cluttered | current | GT-C2 | C2 |
| click_bell | cluttered | initial_current | GT-C3 | C3 |
| click_bell | clean | current | GT-C5 | C5 |
| click_bell | clean | initial_current | GT-C6 | C6 |
| move_can_pot | cluttered | current | GT-C2 | C2 |
| move_can_pot | cluttered | initial_current | GT-C3 | C3 |
| move_can_pot | clean | current | GT-C5 | C5 |
| move_can_pot | clean | initial_current | GT-C6 | C6 |

---

## 注意事項

1. **チェックポイントパス**: 全コマンドで `epoch=0500-*.ckpt` のワイルドカードパターンを使用。トレーニング完了後にbashが自動でマッチする。

2. **VILAサーバー**: 全12条件でVILAサーバーが必要（評価時はVILAパスを使用するため）

3. **評価環境**: 全条件で `--eval_env cluttered` を指定（cluttered環境で評価）

4. **結果出力先**: `eval_results/` ディレクトリに保存される

5. **ログファイル**: 各コマンドの実行ログは `eval_gt_{task}_{env}_{mode}.log` に保存される

---

## トレーニング完了確認コマンド

```bash
find /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_* -name "epoch=0500-*.ckpt" | wc -l
```

12個のファイルが見つかれば全モデルのトレーニング完了。
