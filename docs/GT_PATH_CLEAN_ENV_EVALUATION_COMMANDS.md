# GT Path モデル Clean環境評価コマンド

**作成日**: 2025-01-26

## 概要

GT Path（Ground Truth Path）でトレーニングしたManiFlowモデルを**clean環境**で評価するコマンド一覧。
対象タスク: `beat_block_hammer`

**目的**: GTパスモデルのin-domain性能を確認し、VILAパスモデルとの比較分析を行う

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
   - 起動に約3-4分かかる
   - 「Server READY!」が表示されたら準備完了

---

## 出力ディレクトリ

| 評価種別 | 出力ディレクトリ |
|---------|-----------------|
| VILAパス + cluttered評価 | `eval_results/` |
| VILAパス + clean評価 | `eval_results_clean/` |
| GTパス + cluttered評価 | `eval_results_gt/` |
| **GTパス + clean評価（今回）** | **`eval_results_gt_clean/`** |

---

## チェックポイントベースパス

```
/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/
```

---

## beat_block_hammer

### GT-C2: cluttered学習 + current（VILAあり）→ clean評価

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode current --env cluttered --eval_env clean --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt_clean --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_cluttered_beat_block_hammer/overlay_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_clean_beat_block_hammer_cluttered_current.log 2>&1 &
```

### GT-C3: cluttered学習 + initial_current（Memory Function、VILAあり）→ clean評価

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode initial_current --env cluttered --eval_env clean --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt_clean --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_cluttered_beat_block_hammer/overlay_initial_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_clean_beat_block_hammer_cluttered_initial_current.log 2>&1 &
```

### GT-C5: clean学習 + current（VILAあり）→ clean評価

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode current --env clean --eval_env clean --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt_clean --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_clean_beat_block_hammer/overlay_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_clean_beat_block_hammer_clean_current.log 2>&1 &
```

### GT-C6: clean学習 + initial_current（Memory Function、VILAあり）→ clean評価

```bash
nohup bash -c 'source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && rm -rf ~/.cache/warp && ln -s /gscratch/scrubbed/naoto03/.cache/warp ~/.cache/warp && cd "/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/policy/ManiFlow_HAMSTER" && bash eval.sh --task beat_block_hammer --mode initial_current --env clean --eval_env clean --seed 42 --episodes 100 --vila_port 8000 --output_dir ./eval_results_gt_clean --checkpoint /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_clean_beat_block_hammer/overlay_initial_current_seed42/checkpoints/epoch=0500-*.ckpt' > eval_gt_clean_beat_block_hammer_clean_initial_current.log 2>&1 &
```

---

## 条件マトリクス

| GT条件 | 学習環境 | 学習パス | モード | 評価環境 | VILAサーバー |
|--------|---------|---------|--------|---------|-------------|
| GT-C2 | cluttered | GT Path | current | **clean** | 必要 |
| GT-C3 | cluttered | GT Path | initial_current | **clean** | 必要 |
| GT-C5 | clean | GT Path | current | **clean** | 必要 |
| GT-C6 | clean | GT Path | initial_current | **clean** | 必要 |

---

## 実行順序の推奨

全条件でVILAサーバーが必要なため：

1. **VILAサーバー起動**
2. **全コマンド実行**（GT-C2, GT-C3, GT-C5, GT-C6）
3. **VILAサーバー停止**

---

## 注意事項

1. **チェックポイントパス**: 全コマンドで `epoch=0500-*.ckpt` のワイルドカードパターンを使用

2. **出力先**: `eval_results_gt_clean/` に保存（他の評価結果と混同しない）

3. **ログファイル**: 各コマンドの実行ログは `eval_gt_clean_beat_block_hammer_{env}_{mode}.log` に保存

4. **評価時のパス**: 評価時は**VILAパス**を使用（学習時のみGTパスを使用）

5. **結果ディレクトリ構造**:
   ```
   eval_results_gt_clean/
   └── beat_block_hammer/
       ├── gt_condition2_cluttered_overlay_current_evalclean/
       ├── gt_condition3_cluttered_overlay_initial_current_evalclean/
       ├── gt_condition5_clean_overlay_current_evalclean/
       └── gt_condition6_clean_overlay_initial_current_evalclean/
   ```

---

## 比較分析

評価完了後、以下の比較が可能：

| 比較 | 目的 |
|-----|-----|
| GT-C5 vs C5 (clean評価) | GTパス vs VILAパス学習の効果（in-domain） |
| GT-C2 vs C2 (clean評価) | GTパス vs VILAパス学習の効果（cross-domain） |
| GT-C5 vs GT-C2 (clean評価) | GTパスモデルでの学習環境の影響 |
| GT-C6 vs GT-C5 (clean評価) | Memory Functionの効果（GTパス、in-domain） |

---

## トレーニング完了確認コマンド

```bash
find /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/gt_*beat_block_hammer -name "epoch=0500-*.ckpt" | wc -l
```

4個のファイルが見つかればbeat_block_hammerのGTパスモデルのトレーニング完了。
