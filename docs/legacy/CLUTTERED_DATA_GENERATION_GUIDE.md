# RoboTwin 2.0 Cluttered Table データ生成ガイド

**目的**: 別PCでRoboTwin 2.0のcluttered table（散らかった机上環境）データセットを生成し、HPC環境に転送する

---

## 1. 必要なデータの概要

### 対象タスク（6つのシングルアームタスク）

| タスク名 | 説明 |
|---------|------|
| `beat_block_hammer` | ハンマーでブロックを叩く |
| `click_bell` | ベルを押す |
| `move_can_pot` | 缶を鍋に移動 |
| `open_microwave` | 電子レンジを開ける |
| `place_object_stand` | オブジェクトをスタンドに置く |
| `turn_switch` | スイッチを回す |

### データ量
- **各タスク**: 50エピソード
- **合計**: 6タスク × 50エピソード = 300エピソード

---

## 2. 環境セットアップ

### 2.1 リポジトリのクローン

```bash
# ManiFlow Policy (robotwin2.0ブランチ)
git clone -b robotwin2.0 https://github.com/geyan21/ManiFlow_Policy.git
cd ManiFlow_Policy/RoboTwin
```

### 2.2 Docker環境の構築

メンターのDockerfileを使用:

```bash
cd policy/ManiFlow/Docker
./BUILD_DOCKER_IMAGE.sh
```

### 2.3 Curoboのインストール（コンテナ内で実行）

```bash
cd envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../..
```

### 2.4 必要なアセットのダウンロード

HuggingFaceから以下をダウンロード:
- `embodiments.zip` → `assets/embodiments/`に展開
- `objects.zip` → `assets/objects/`に展開

```bash
# HuggingFace CLIを使用する場合
huggingface-cli download TonyStar/RoboTwin2 embodiments.zip --local-dir ./
huggingface-cli download TonyStar/RoboTwin2 objects.zip --local-dir ./
unzip embodiments.zip -d assets/
unzip objects.zip -d assets/
```

---

## 3. 設定ファイル

### 3.1 タスク設定ファイルの作成

`task_config/aloha-agilex_cluttered_50.yml` として以下を作成:

```yaml
# Cluttered table configuration for single-arm tasks
render_freq: 0
episode_num: 50
use_seed: false
save_freq: 15
embodiment: [aloha-agilex]
language_num: 100
domain_randomization:
  random_background: false
  cluttered_table: true          # ← これが重要！
  clean_background_rate: 1
  random_head_camera_dis: 0
  random_table_height: 0
  random_light: false
  crazy_random_light_rate: 0
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  third_view: false
  depth: false
  pointcloud: false
  observer: false
  endpose: true
  qpos: true
  mesh_segmentation: false
  actor_segmentation: false
pcd_down_sample_num: 1024
pcd_crop: true
save_path: ./dataset/dataset
clear_cache_freq: 5
collect_data: true
eval_video_log: true
```

### 3.2 重要な設定項目

| 設定 | 値 | 説明 |
|-----|-----|------|
| `cluttered_table` | `true` | 机上にdistractor objectsを配置 |
| `episode_num` | `50` | 各タスク50エピソード |
| `embodiment` | `aloha-agilex` | 使用するロボット |
| `rgb` | `true` | RGBカメラ画像を収集 |
| `endpose` | `true` | エンドエフェクタの姿勢を収集 |
| `qpos` | `true` | 関節角度を収集 |

---

## 4. データ収集の実行

### 4.1 単一タスクの実行

```bash
python script/collect_data.py <task_name> aloha-agilex_cluttered_50
```

### 4.2 全6タスクの一括実行スクリプト

```bash
#!/bin/bash
TASKS=(
    "beat_block_hammer"
    "click_bell"
    "move_can_pot"
    "open_microwave"
    "place_object_stand"
    "turn_switch"
)

CONFIG="aloha-agilex_cluttered_50"

for task in "${TASKS[@]}"; do
    echo "=========================================="
    echo "Collecting data for: $task"
    echo "=========================================="
    python script/collect_data.py "$task" "$CONFIG"
done

echo "All tasks completed!"
```

---

## 5. 期待されるデータ構造

### 5.1 出力ディレクトリ構造

```
dataset/dataset/
├── beat_block_hammer/
│   ├── episode_0/
│   │   ├── head_camera/
│   │   │   ├── rgb/
│   │   │   │   ├── 0.png
│   │   │   │   ├── 1.png
│   │   │   │   └── ...
│   │   ├── wrist_camera/
│   │   │   ├── rgb/
│   │   │   │   └── ...
│   │   ├── endpose.npy       # エンドエフェクタ姿勢
│   │   ├── qpos.npy          # 関節角度
│   │   └── metadata.json     # エピソードメタデータ
│   ├── episode_1/
│   ├── ...
│   └── episode_49/
├── click_bell/
│   └── ...
├── move_can_pot/
│   └── ...
├── open_microwave/
│   └── ...
├── place_object_stand/
│   └── ...
└── turn_switch/
    └── ...
```

### 5.2 各ファイルの内容

| ファイル | 形式 | 内容 |
|---------|------|------|
| `rgb/*.png` | PNG画像 | 640x480 RGB画像 |
| `endpose.npy` | NumPy配列 | shape: (T, 7) - [x, y, z, qx, qy, qz, qw] |
| `qpos.npy` | NumPy配列 | shape: (T, N) - 関節角度 |
| `metadata.json` | JSON | タスク情報、成功/失敗フラグなど |

---

## 6. データの転送

### 6.1 圧縮

```bash
cd dataset
tar -czvf robotwin2_cluttered_6tasks_50eps.tar.gz dataset/
```

### 6.2 転送先

HPC環境の以下のパスに配置:
```
/mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/dataset/
```

### 6.3 転送方法（例）

```bash
# SCPの場合
scp robotwin2_cluttered_6tasks_50eps.tar.gz user@hpc:/path/to/destination/

# rsyncの場合
rsync -avz --progress robotwin2_cluttered_6tasks_50eps.tar.gz user@hpc:/path/to/destination/
```

---

## 7. 検証チェックリスト

データ生成後、以下を確認:

- [ ] 各タスクに50エピソードが存在する
- [ ] 各エピソードにhead_camera/rgb/とwrist_camera/rgb/が存在する
- [ ] endpose.npyとqpos.npyが各エピソードに存在する
- [ ] 画像が正しく保存されている（破損なし）
- [ ] cluttered_tableが有効になっている（画像に distractor objects が映っている）

### 簡易検証コマンド

```bash
# エピソード数の確認
for task in beat_block_hammer click_bell move_can_pot open_microwave place_object_stand turn_switch; do
    echo "$task: $(ls -d dataset/dataset/$task/episode_* 2>/dev/null | wc -l) episodes"
done

# ファイル存在確認
ls dataset/dataset/beat_block_hammer/episode_0/
```

---

## 8. トラブルシューティング

### Curobo関連エラー

- `planner: "curobo"` を `assets/embodiments/aloha-agilex/config.yml` で設定
- JITコンパイルに時間がかかる（初回のみ）

### メモリ不足

- `render_freq: 0` でレンダリング頻度を下げる
- `clear_cache_freq: 5` でキャッシュを頻繁にクリア

### GPU関連

- NVIDIA GPUが必要（CUDA 12.1推奨）
- `--gpus all` でDockerコンテナにGPUをパススルー

---

## 9. 参考リンク

- [ManiFlow Policy (robotwin2.0)](https://github.com/geyan21/ManiFlow_Policy/tree/robotwin2.0)
- [RoboTwin 2.0 HuggingFace](https://huggingface.co/TonyStar/RoboTwin2)
- [Curobo (NVIDIA)](https://github.com/NVlabs/curobo)

---

## 10. 連絡事項

データ生成完了後、以下を報告してください:
1. 生成したエピソード数（タスクごと）
2. 総データサイズ
3. 転送方法の希望（SCP/rsync/その他）
