# RoboTwin 2.0 Cluttered Table データ生成

**最終更新**: 2025-12-16

## 目標

RoboTwin 2.0で**cluttered table（散らかった机上環境）**のデータセットを生成する。

- **対象タスク**: 6つのシングルアームタスク
  - beat_block_hammer
  - click_bell
  - move_can_pot
  - open_microwave
  - place_object_stand
  - turn_switch
- **エピソード数**: 各タスク50エピソード
- **設定変更**: ManiFlow公式RoboTwin2.0設定をベースに `cluttered_table: true` のみ変更

---

## 現在の状況（2025-12-16）

### 結論

**HPC Singularity環境ではCuroboがSegfaultで動作しない** → **別PCでDockerを使用してデータ生成することを決定**

詳細なガイドは [CLUTTERED_DATA_GENERATION_GUIDE.md](./CLUTTERED_DATA_GENERATION_GUIDE.md) を参照。

### 完了したこと

1. **設定ファイル作成**
   - `RoboTwin2.0/task_config/aloha-agilex_cluttered_50.yml` - 本番用（50エピソード）
   - `RoboTwin2.0/task_config/aloha-agilex_cluttered_test.yml` - テスト用（1エピソード）

2. **データ収集スクリプト作成**
   - `RoboTwin2.0/collect_cluttered_6tasks_singularity.sh` - 6タスク一括実行

3. **Objaverseアセットダウンロード**
   - HuggingFaceから`objects.zip`をダウンロードして展開
   - `assets/objects/objaverse/list.json` 含む

4. **Embodiment設定追加**
   - `embodiments.zip`から`config.yml`を取得
   - `assets/embodiments/aloha-agilex/config.yml`

5. **Curoboインストール完了**
   - `third_party/curobo/`にクローン
   - `pip install -e . --no-build-isolation --target=/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages`
   - インポートテスト成功: `from curobo.types.math import Pose as CuroboPose`

6. **Curobo設定ファイル修正**
   - `curobo_left_tmp.yml` → `curobo_left.yml` にリネーム
   - `curobo_right_tmp.yml` → `curobo_right.yml` にリネーム
   - `${ASSETS_PATH}` プレースホルダーを絶対パスに置換

7. **config.yml更新**
   - `planner: "curobo"` に設定済み

8. **別PC向けデータ生成ガイド作成**
   - `docs/CLUTTERED_DATA_GENERATION_GUIDE.md`

---

## 直面した問題と解決

### 問題1: mplib.Planner() Segfault

**症状**: RoboTwin 2.0のデータ収集実行時にSegmentation fault発生

```
Segmentation fault (core dumped)
```

**原因**: `mplib.Planner()`の初期化時にクラッシュ。Singularity環境とmplibの互換性問題。

**試した解決策**:
1. `planner.py`でCuroboPlanner importエラー時のフォールバック追加
2. `robot.py`でMplibPlannerへのフォールバックロジック追加
3. `config.yml`を`planner: "mplib_screw"`に変更

**結果**: mplibも同様にSegfault → **Curoboインストールで回避を試みた**

### 問題2: Curoboインストール時のディスククォータエラー

**症状**:
```
ERROR: Could not install packages due to an OSError: [Errno 122] Disk quota exceeded
```

**原因**: pipがデフォルトで`~/.local/`にインストールしようとし、ホームディレクトリのクォータを超過

**解決策**: `--target`と`--cache-dir`オプションで`/gscratch/scrubbed`に直接インストール

```bash
pip install -e /path/to/curobo --no-build-isolation \
    --target=/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages \
    --cache-dir=/gscratch/scrubbed/naoto03/.cache/pip
```

### 問題3: Curobo設定ファイルの命名問題

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../curobo_left.yml'
```

**原因**: HuggingFaceの`embodiments.zip`に含まれるファイルが`curobo_left_tmp.yml`という名前になっていた

**解決策**: ファイルをリネーム
```bash
cp curobo_left_tmp.yml curobo_left.yml
cp curobo_right_tmp.yml curobo_right.yml
```

### 問題4: Curobo設定ファイルのパスプレースホルダー

**症状**:
```
FileNotFoundError: No such file or directory: '${ASSETS_PATH}/assets/embodiments/...'
```

**原因**: yml内の`${ASSETS_PATH}`が環境変数として展開されず文字列のまま使用された

**解決策**: 絶対パスに置換
```yaml
# Before
urdf_path: ${ASSETS_PATH}/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf

# After
urdf_path: /mmfs1/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
```

### 問題5: Curobo JIT後のSegfault（未解決）

**症状**: CuroboのJITコンパイル（kinematics_fused_cu, geom_cu, tensor_step_cu, lbfgs_step_cu, line_search_cu）は成功するが、その後Segfaultが発生

```
kinematics_fused_cu not found, JIT compiling...
geom_cu binary not found, jit compiling...
tensor_step_cu not found, jit compiling...
lbfgs_step_cu not found, JIT compiling...
line_search_cu not found, JIT compiling...
Segmentation fault (core dumped)
```

**試した解決策**:
1. `torch_extensions`キャッシュをクリアして再実行 → 同様にSegfault
2. 複数回の再試行 → 同様の結果

**結論**: Singularity環境とCuroboの根本的な互換性問題。Dockerを使用する別PCでデータ生成を行う

---

## 次回やること

### 別PCでのデータ生成

[CLUTTERED_DATA_GENERATION_GUIDE.md](./CLUTTERED_DATA_GENERATION_GUIDE.md) に従って:

1. Dockerで環境構築
2. 6タスク × 50エピソードのデータ生成
3. データをHPC環境に転送

---

## ファイル構成

### 設定ファイル

| ファイル | 説明 |
|---------|------|
| `task_config/aloha-agilex_cluttered_50.yml` | 本番用（50エピソード、cluttered_table: true） |
| `task_config/aloha-agilex_cluttered_test.yml` | テスト用（1エピソード） |
| `assets/embodiments/aloha-agilex/config.yml` | ロボット設定（planner: curobo） |
| `assets/embodiments/aloha-agilex/curobo_left.yml` | 左腕Curobo設定（絶対パス使用） |
| `assets/embodiments/aloha-agilex/curobo_right.yml` | 右腕Curobo設定（絶対パス使用） |

### 修正したコード

| ファイル | 修正内容 |
|---------|---------|
| `envs/robot/planner.py` | CuroboPlanner importエラー時に`None`を設定 |
| `envs/robot/robot.py` | MplibPlannerフォールバックロジック追加 |

### インストール済みパッケージ（HPC環境）

| パッケージ | 場所 |
|-----------|------|
| curobo（ソース） | `third_party/curobo/src/curobo/` |
| curobo依存関係 | `/gscratch/scrubbed/naoto03/.local/lib/python3.10/site-packages/` |

---

## 参考情報

### メンターからの情報

curoboインストールの公式手順（ManiFlow Docker）：
- Dockerfile: https://github.com/geyan21/ManiFlow_Policy/blob/robotwin2.0/RoboTwin/policy/ManiFlow/Docker/Dockerfile
- 初期化スクリプト: https://github.com/geyan21/ManiFlow_Policy/blob/robotwin2.0/RoboTwin/policy/ManiFlow/Docker/initialize-docker-container.sh

インストールコマンド（メンター版）：
```bash
cd envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
```

**注意**: Dockerイメージはローカルビルドのみで、レジストリには公開されていない。
イメージ名: `maniflow_robotwin2_docker:latest`

### 依存関係の警告（無視してOK）

curoboインストール時に以下の警告が出るが、RoboTwin 2.0での使用には影響なし：

```
moviepy 2.2.1 requires pillow<12.0,>=9.2.0, but you have pillow 12.0.0
mplib 0.2.1 requires numpy<2.0, but you have numpy 2.2.6
torchaudio 2.5.1+cu121 requires torch==2.5.1, but you have torch 2.9.1
torchvision 0.20.1+cu121 requires torch==2.5.1, but you have torch 2.9.1
```

---

## 関連ドキュメント

- [CLUTTERED_DATA_GENERATION_GUIDE.md](./CLUTTERED_DATA_GENERATION_GUIDE.md) - **別PCでのデータ生成ガイド（推奨）**
- [NEXT_PHASE_REQUIREMENTS.md](./NEXT_PHASE_REQUIREMENTS.md) - ManiFlow + パス入力学習の計画
- [ROBOTWIN_OVERVIEW.md](./ROBOTWIN_OVERVIEW.md) - RoboTwinの概要
