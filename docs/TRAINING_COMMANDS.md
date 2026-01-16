# トレーニングコマンド一覧

## 概要

5タスク × 6条件 = 30個のトレーニングコマンド

### タスク一覧
- click_bell
- move_can_pot
- open_microwave
- turn_switch
- adjust_bottle

### 条件一覧
| 条件 | 環境 | モデル | スクリプト |
|------|------|--------|-----------|
| 1 | cluttered | original | train_original.sh |
| 2 | cluttered | overlay_current | train_overlay_current.sh |
| 3 | cluttered | overlay_initial_current | train_overlay_initial_current.sh |
| 4 | clean | original | train_original.sh |
| 5 | clean | overlay_current | train_overlay_current.sh |
| 6 | clean | overlay_initial_current | train_overlay_initial_current.sh |

---

## click_bell

### 条件1: cluttered + original
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh cluttered click_bell 0 42 > /dev/null 2>&1 &
```

### 条件2: cluttered + overlay_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh cluttered click_bell 0 42 > /dev/null 2>&1 &
```

### 条件3: cluttered + overlay_initial_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh cluttered click_bell 0 42 > /dev/null 2>&1 &
```

### 条件4: clean + original
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh clean click_bell 0 42 > /dev/null 2>&1 &
```

### 条件5: clean + overlay_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh clean click_bell 0 42 > /dev/null 2>&1 &
```

### 条件6: clean + overlay_initial_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh clean click_bell 0 42 > /dev/null 2>&1 &
```

---

## move_can_pot

### 条件1: cluttered + original
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh cluttered move_can_pot 0 42 > /dev/null 2>&1 &
```

### 条件2: cluttered + overlay_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh cluttered move_can_pot 0 42 > /dev/null 2>&1 &
```

### 条件3: cluttered + overlay_initial_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh cluttered move_can_pot 0 42 > /dev/null 2>&1 &
```

### 条件4: clean + original
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh clean move_can_pot 0 42 > /dev/null 2>&1 &
```

### 条件5: clean + overlay_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh clean move_can_pot 0 42 > /dev/null 2>&1 &
```

### 条件6: clean + overlay_initial_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh clean move_can_pot 0 42 > /dev/null 2>&1 &
```

---

## open_microwave

### 条件1: cluttered + original
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh cluttered open_microwave 0 42 > /dev/null 2>&1 &
```

### 条件2: cluttered + overlay_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh cluttered open_microwave 0 42 > /dev/null 2>&1 &
```

### 条件3: cluttered + overlay_initial_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh cluttered open_microwave 0 42 > /dev/null 2>&1 &
```

### 条件4: clean + original
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh clean open_microwave 0 42 > /dev/null 2>&1 &
```

### 条件5: clean + overlay_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh clean open_microwave 0 42 > /dev/null 2>&1 &
```

### 条件6: clean + overlay_initial_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh clean open_microwave 0 42 > /dev/null 2>&1 &
```

---

## turn_switch

### 条件1: cluttered + original
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh cluttered turn_switch 0 42 > /dev/null 2>&1 &
```

### 条件2: cluttered + overlay_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh cluttered turn_switch 0 42 > /dev/null 2>&1 &
```

### 条件3: cluttered + overlay_initial_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh cluttered turn_switch 0 42 > /dev/null 2>&1 &
```

### 条件4: clean + original
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh clean turn_switch 0 42 > /dev/null 2>&1 &
```

### 条件5: clean + overlay_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh clean turn_switch 0 42 > /dev/null 2>&1 &
```

### 条件6: clean + overlay_initial_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh clean turn_switch 0 42 > /dev/null 2>&1 &
```

---

## adjust_bottle

### 条件1: cluttered + original
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh cluttered adjust_bottle 0 42 > /dev/null 2>&1 &
```

### 条件2: cluttered + overlay_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh cluttered adjust_bottle 0 42 > /dev/null 2>&1 &
```

### 条件3: cluttered + overlay_initial_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh cluttered adjust_bottle 0 42 > /dev/null 2>&1 &
```

### 条件4: clean + original
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_original.sh clean adjust_bottle 0 42 > /dev/null 2>&1 &
```

### 条件5: clean + overlay_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_current.sh clean adjust_bottle 0 42 > /dev/null 2>&1 &
```

### 条件6: clean + overlay_initial_current
```bash
source /gscratch/scrubbed/naoto03/miniconda3/etc/profile.d/conda.sh && conda activate robotwin && module load cuda/12.4.1 gcc/13.2.0 && cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/scripts && WANDB_MODE=disabled nohup ./train_overlay_initial_current.sh clean adjust_bottle 0 42 > /dev/null 2>&1 &
```

---

## 前提条件

トレーニングを開始する前に、各タスクのZarrファイルが生成されている必要がある。

### 必要なZarrファイル（タスクごと）

| タスク | clean_original | clean_overlay | cluttered_original | cluttered_overlay |
|--------|----------------|---------------|--------------------|--------------------|
| click_bell | 必要 | 必要 | 必要 | 必要 |
| move_can_pot | 必要 | 必要 | 必要 | 必要 |
| open_microwave | 必要 | 必要 | 必要 | 必要 |
| turn_switch | 必要 | 必要 | 必要 | 必要 |
| adjust_bottle | 必要 | 必要 | 必要 | 必要 |

### Zarrファイル生成コマンド

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow

# clean環境
python scripts/convert_original_to_zarr.py --input-dir /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/evaluation_tasks_clean --output data/zarr/clean_original_{task}.zarr --tasks {task} --episodes 50

python scripts/convert_overlay_to_zarr.py --overlay-base /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/evaluation_tasks_clean --output data/zarr/clean_overlay_{task}.zarr --tasks {task} --episodes 50

# cluttered環境
python scripts/convert_original_to_zarr.py --input-dir /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/evaluation_tasks_cluttered --output data/zarr/cluttered_original_{task}.zarr --tasks {task} --episodes 50

python scripts/convert_overlay_to_zarr.py --overlay-base /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results/evaluation_tasks_cluttered --output data/zarr/cluttered_overlay_{task}.zarr --tasks {task} --episodes 50
```

---

## ログ確認

```bash
# リアルタイムでログを確認
tail -f /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/logs/train_*.log

# プロセス確認
ps aux | grep train

# GPU使用状況
nvidia-smi
```

---

## 出力ディレクトリ

トレーニング結果は以下に保存される:

```
/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/ManiFlow/data/outputs/
├── {env}_{task}/
│   ├── original_seed42/
│   │   ├── checkpoints/
│   │   │   ├── latest.ckpt
│   │   │   └── epoch=XXXX-val_loss=X.XXXXXX.ckpt
│   │   └── training_metrics.jsonl  # 構造化ログ
│   ├── overlay_current_seed42/
│   └── overlay_initial_current_seed42/
```
