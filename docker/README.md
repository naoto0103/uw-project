# HAMSTER-ManiFlow Docker for Hyak

Hyak HPC環境でHAMSTER-ManiFlow統合プロジェクトを実行するためのDockerセットアップ。

## ファイル構成

```
docker/
├── Dockerfile              # Dockerイメージ定義
├── requirements_docker.txt # Python依存関係
├── hyak_setup.sh          # Hyak初期セットアップスクリプト
├── initialize_container.sh # コンテナ内初期化スクリプト
└── README.md              # このファイル
```

## クイックスタート

### 1. ローカルでDockerイメージをビルド

```bash
cd docker/
docker build -t naototo0103/hamster-maniflow:latest .
docker push naototo0103/hamster-maniflow:latest
```

### 2. HyakでGPUノードを取得

```bash
# A40 GPUの場合
srun -p gpu-a40 -A {lab_account} --nodes=1 --cpus-per-task=32 \
     --mem=400G --time=168:00:00 --gpus=2 --pty /bin/bash

# L40s GPUの場合
srun -p gpu-l40s -A cse --nodes=1 --cpus-per-task=120 \
     --mem=1000G --time=24:00:00 --gpus=6 --pty /bin/bash
```

### 3. Singularityイメージをpull

```bash
# 環境変数設定
export SINGULARITY_CACHEDIR=/gscratch/scrubbed/${USER}/singularity/cache
export SINGULARITY_TMPDIR=/gscratch/scrubbed/${USER}/singularity/tmp
export APPTAINER_CACHEDIR=/gscratch/scrubbed/${USER}/singularity/cache

# ディレクトリ作成
mkdir -p ${SINGULARITY_CACHEDIR} ${SINGULARITY_TMPDIR}
mkdir -p /gscratch/scrubbed/${USER}/singularity/images

# モジュールロード & pull
module load singularity
cd /gscratch/scrubbed/${USER}/singularity/images
singularity pull docker://naototo0103/hamster-maniflow:latest
```

### 4. Singularityインスタンス起動

```bash
# インスタンス作成
singularity instance start --nv \
    --bind /gscratch/:/gscratch/:rw \
    hamster-maniflow_latest.sif hamster_train

# コンテナに入る
singularity shell instance://hamster_train
```

### 5. コンテナ内で初期化

```bash
# リポジトリクローン
cd /gscratch/scrubbed/${USER}/code
git clone https://github.com/YOUR_REPO/HAMSTER-ManiFlow-Integration.git
cd HAMSTER-ManiFlow-Integration

# 初期化スクリプト実行
bash docker/initialize_container.sh
```

## データ転送

```bash
# ローカルPCから（別ターミナルで実行）
rsync -avzP /path/to/local/data/ hyak:/gscratch/scrubbed/${USER}/data/

# 必要なデータ:
# - ManiFlow/data/*.zarr (RoboTwinデータ)
# - ManiFlow/data/*/hamster_paths.pkl (HAMSTERパス)
```

## 使用方法

### Qwen3-VLサーバー起動

```bash
cd /gscratch/scrubbed/${USER}/code/HAMSTER-ManiFlow-Integration
python HAMSTER/server_qwen3.py --port 8001
```

### ManiFlowトレーニング

```bash
cd ManiFlow
bash scripts/train_eval_hamster_maniflow.sh
```

## トラブルシューティング

### PyTorch3Dのインポートエラー

```bash
# 再インストール
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### MuJoCoのエラー

```bash
# 環境変数確認
echo $LD_LIBRARY_PATH
# /root/.mujoco/mujoco210/bin が含まれているか確認
```

### HuggingFaceモデルのダウンロード先

```bash
# キャッシュディレクトリを確認
echo $HF_HOME
# /workspace/cache/huggingface または /gscratch/scrubbed/${USER}/cache/huggingface
```

## 注意事項

- **モデルサイズ**: Qwen3-VL-8Bは約17GB、初回起動時に自動ダウンロード
- **ストレージ**: /gscratch/scrubbed/ は90日で自動削除されるので注意
- **GPU**: A40 (48GB) / L40s (48GB) どちらでも動作
