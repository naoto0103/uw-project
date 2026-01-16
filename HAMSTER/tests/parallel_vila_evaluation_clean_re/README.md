# Parallel VILA Path Generation

4つのGPUを使用してVILAパス生成を並列実行するためのスクリプト群。

## ディレクトリ構成

```
parallel_vila_evaluation_clean/
├── README.md           # このファイル
├── config.py           # 設定（ポート、パス、タスク一覧）
├── start_servers.sh    # 4つのVILAサーバーを起動
├── stop_servers.sh     # 全サーバーを停止
├── generate_paths.py   # 並列パス生成メインスクリプト
└── logs/               # サーバーログ（自動生成）
```

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│  メインプロセス                                          │
│  ┌─────────────┐                                        │
│  │ フレームキュー │ ← 全54,572フレームを投入             │
│  └──────┬──────┘                                        │
│         │                                               │
│    ┌────┴────┬────────┬────────┐                       │
│    ▼         ▼        ▼        ▼                       │
│ Worker 0  Worker 1  Worker 2  Worker 3                 │
│ (GPU 0)   (GPU 1)   (GPU 2)   (GPU 3)                  │
│ port:8000 port:8001 port:8002 port:8003                │
└─────────────────────────────────────────────────────────┘
```

動的ディスパッチ方式：
- 各ワーカーは空いたら次のフレームを取得
- 処理時間のばらつきを自動吸収
- GPU使用率を最大化

## 使い方

### 1. GPUノードの確保

```bash
srun -p gpu-a40 -A escience --nodes=1 --cpus-per-task=32 \
     --mem=400G --time=24:00:00 --gpus=4 --pty /bin/bash
```

### 2. VILAサーバーの起動

```bash
cd /gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/tests/parallel_vila_evaluation_clean
./start_servers.sh
```

起動完了まで約3分かかる（各GPUにモデルをロード）。
全サーバーが "READY" になるまで待つ。

### 3. パス生成の実行

```bash
# Singularity内で実行（--bindオプション必須）
/usr/bin/singularity exec \
    --bind /gscratch/:/gscratch/:rw \
    --bind /mmfs1/:/mmfs1/:rw \
    --env PYTHONPATH="/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages-vila" \
    --env SSL_CERT_FILE="" \
    --env SSL_CERT_DIR="" \
    --env REQUESTS_CA_BUNDLE="" \
    /gscratch/scrubbed/naoto03/singularity/hamster-maniflow_latest.sif \
    python generate_paths.py --episodes 50
```

**注意**: `--bind`オプションは必須。これがないとデータディレクトリやモデルにアクセスできない。

または、サーバーと同じノード内でSingularity外から直接実行も可能：
```bash
python generate_paths.py --episodes 50
```

### 4. サーバーの停止

```bash
./stop_servers.sh
```

## コマンドラインオプション

### generate_paths.py

```
--tasks TASK [TASK ...]   処理するタスク（デフォルト: 全6タスク）
--episodes N              エピソード数（デフォルト: 50）
--num-gpus N              使用するGPU数（デフォルト: 4）
--results-dir PATH        結果ディレクトリ
--skip-server-check       サーバーチェックをスキップ
```

例：特定のタスクのみ処理
```bash
python generate_paths.py --tasks beat_block_hammer click_bell --episodes 10
```

## 出力

```
HAMSTER/results/evaluation_tasks_clean/
├── {task_name}/
│   └── episode_{00-49}/
│       ├── frames/           # 入力フレーム（既存）
│       ├── paths/            # 生成されたパス (.pkl)
│       └── raw_outputs/      # モデルの生出力 (.txt)
└── parallel_generation_summary.json  # 実行サマリー
```

## 推定処理時間

| 構成 | フレーム数 | 推定時間 |
|------|-----------|---------|
| 1 GPU | 54,572 | 約27時間 |
| 4 GPU | 54,572 | 約7時間 |

## トラブルシューティング

### サーバーが起動しない

```bash
# ログを確認
cat logs/server_gpu0_port8000.log
```

よくある原因：
- VRAM不足（各サーバーに約26GB必要）
- モデルパスが間違っている
- Singularityモジュールがロードされていない

### 処理が遅い

- サーバーログでエラーを確認
- 特定のワーカーだけ遅い場合、そのGPUに問題がある可能性

### 中断からの再開

スクリプトは既に処理済みのフレームをスキップするので、
そのまま再実行すれば続きから処理される。

```bash
# 中断後、そのまま再実行
python generate_paths.py --episodes 50
```

## 注意事項

- 4つのVILAサーバーで約104GB VRAM使用（26GB × 4）
- サーバー起動中は他のGPU処理ができない
- 長時間実行の場合、tmuxやscreenの使用を推奨
