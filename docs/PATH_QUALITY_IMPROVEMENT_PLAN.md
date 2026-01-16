# Path Quality Improvement Plan

## Date: 2025-01-05

## Background

メンターとのミーティングで、現在の実験結果（Memory Functionの失敗、In-domain改善なし）の根本原因として、**VLMのパス生成品質**に問題がある可能性が指摘された。

---

## Problem Statement

### Core Issue

HAMSTERで使用されているVLM（ファインチューニングされたVILA-1.5-13B）は、**タスクの最初のフレーム（Frame 0）のみ**でファインチューニングされている。

そのため、タスク実行中のフレーム（Frame 16, 32, ...）に対しては正確なパス生成ができない可能性がある。

### Impact on Current Results

| 現象 | パス品質問題による説明 |
|------|------------------------|
| Memory Function（Initial + Current）の失敗 | Initial path（Frame 0）は高精度、Current path（実行中）は低精度 → 異なる精度の情報を混ぜて入力 → モデルが混乱 |
| In-domain改善なし | 訓練データ自体のパス精度が悪い → ManiFlowが「ノイズの多いパス」で学習 → パス入力が役に立たない |
| Cross-domainでのみ効果あり（C5: +7.7%） | パス精度は低くても、VILAのセマンティック理解が大まかな方向性を提供 → ドメインギャップの橋渡しに寄与 |

---

## Next Actions

### Step 1: Visualize Paths in Training Data

**目的**: VILAが生成した訓練データのパス精度を目視確認

**確認項目**:
- [ ] Frame 0でのパス精度
- [ ] Frame 16, 32, ... でのパス精度
- [ ] パス生成失敗率（fallbackが発生した頻度）
- [ ] グリッパー状態変化点の正確性

**出力**: 各タスク・各条件でのパスオーバーレイ画像サンプル

### Step 2: Visualize Paths in Evaluation

**目的**: 評価時（実行中）のパス精度を目視確認

**確認項目**:
- [ ] エピソード進行に伴うパス品質の変化
- [ ] オクルージョン発生時のパス品質
- [ ] 成功エピソードと失敗エピソードでのパス品質比較

**出力**: 評価エピソードのパス推移動画またはフレームごとの画像

### Step 3: Compare and Analyze

**目的**: 訓練データと評価時のパス品質を比較し、問題の程度を定量化

**分析項目**:
- [ ] Frame 0 vs Frame N でのパス品質差
- [ ] 訓練時 vs 評価時でのパス品質差
- [ ] タスクごとのパス品質傾向
- [ ] Ground Truth軌跡との乖離度（可能であれば定量評価）

---

## Improvement Options

### Option A: Ground Truth + Noise (HAMSTER Method)

HAMSTERの論文で実際に使用された手法。

**アプローチ**:
```
訓練データのGround Truth軌跡（3Dエンドエフェクター位置）
    ↓
カメラパラメータを用いて2D投影
    ↓
ガウシアンノイズを追加（VILAの予測誤差をシミュレート）
    ↓
オーバーレイ画像生成
    ↓
ManiFlowの訓練に使用
```

**メリット**:
- 訓練データのパス精度が保証される
- ManiFlowが「正しいパス情報を活用する」ことを学習できる
- 評価時のVILAパス誤差への頑健性もノイズで獲得

**デメリット**:
- 評価時は依然としてVILAのパスを使用するため、訓練と評価でパス品質のギャップが残る
- ノイズレベルの調整が必要

**実装タスク**:
- [ ] RoboTwin 2.0からGround Truth軌跡を抽出
- [ ] 3D→2D投影の実装（カメラパラメータ取得）
- [ ] ノイズ追加ロジックの実装
- [ ] 新しいデータセットクラスの作成または既存クラスの修正

### Option B: Use PEEK's VLM (VILA-1.5-3B)

PEEKで使用されているVLM（ファインチューニングされたVILA-1.5-3B）は、動作中の状態も含めたデータでファインチューニングされている。

**メリット**:
- 根本的にVLMのパス生成能力が向上
- 実行中のフレームでも信頼できるパスが得られる
- 訓練と評価で一貫したパス品質が期待できる

**デメリット**:
- PEEKのモデルウェイトへのアクセスが必要
- モデルサイズが3Bなので推論は速いが、13Bと比較して能力が劣る可能性
- 新しいVLMへの移行作業が必要

**実装タスク**:
- [ ] PEEKのモデルウェイト取得
- [ ] PEEKのVLM推論コードの統合
- [ ] 既存パイプラインとの互換性確認

---

## Decision Criteria

Step 1-3の可視化・分析結果に基づき、以下の基準で方針を決定：

| パス品質の状況 | 推奨アクション |
|----------------|----------------|
| Frame 0は良好、Frame N以降で急激に悪化 | Option A（GT + Noise）を優先検討 |
| 全体的にパス品質が低い | Option B（PEEK VLM）を優先検討 |
| パス品質は問題なし | 別の原因を調査（モデルアーキテクチャ、訓練設定など） |

---

## Path Quality Analysis Results (2025-01-05)

### 可視化方法

- 16フレーム間隔でオーバーレイ画像をサンプリングし、グリッド画像として並べて目視確認
- パス生成失敗フレーム（対応する.pklファイルがないフレーム）には赤い×マークを付与
- 出力先: `analysis/path_quality/grids/{clean,cluttered}/{task}/`

### パス生成失敗数（16フレーム間隔サンプリング）

| Condition | Task | Failed Frames |
|-----------|------|---------------|
| Clean | click_bell | 1 |
| Clean | move_can_pot | 2 |
| Clean | beat_block_hammer | 0 |
| Cluttered | click_bell | 8 |
| Cluttered | move_can_pot | 14 |
| Cluttered | beat_block_hammer | 3 |

### 目視確認による所感

#### Clean条件

**beat_block_hammer**:
- 案外綺麗にパスは作れている
- アームが画像内に映り込んでいる時も、ちゃんとブロックとハンマーの場所にパスの軌跡が描写されている
- 見当違いなパスは基本的にあまり見当たらない
- パスの開始点が、ハンマーの柄の部分ではなく、頭の部分になっていることがほとんど

**click_bell**:
- cluttered・beat_block_hammerより見当違いなパスは少ない印象
- 基本的にベルの真上に短くパスが生成される
- たまにアームの影に対して誤認したり、オクルージョン発生時に全然関係のない場所にパスができたりする場合がある
- ただしそのようなバグっているパスの数は少ない
- タスクの性質上パスが非常に短く、グリッパーの開閉状態を表す円が重なっていたり、片方だけしかなかったり、それも閉じる開くの両方のパターンがあるので、その点がわかりづらい

**move_can_pot**:
- 終点位置が、potの位置に対して、potの中央部分やpotのcanに近い側、potの正面など、少しノイジーだが、基本的にpot周辺には落ち着いている
- 始点位置は、基本的にcanに重なっている印象だが、たまにcanが画面外に出てしまっているときはパスが迷っているような感じ
- アームが缶をつかんでいる時などに、canの位置を誤認しているケースがちらほら見受けられる
- 基本的にcanからpotの方向に向かってピックアンドプレースをするというパスの向きとグリッパーの開閉動作に関しては正しいパスが描写されている

#### Cluttered条件

**beat_block_hammer**:
- とんちんかんなパスになっていることがしばしばある
- アームの陰を対象物だと認識したり、画像内の他の物体を対象物だと誤認しているケースが見られる
- オクルージョンが発生している時の誤認が激しい
- 初期のフレームは比較的対象オブジェクトに対して正しいパスを生成できている場合が多いが、タスクが進むにつれて、見当違いなパスになるケースが見られる

**click_bell**:
- 最初から最後まで終始綺麗なパス（ベルの上だけに短くパスが描写されている状態）が生成されている場合もあれば、全然見当違いなパスが生成されているケースもある
- 見当違いなパスが生成されているときは、同じエピソードの中で複数フレームにおかしなパスが描写されているので、clutteredの特定の環境に対してパス生成がうまくいっていない可能性
- そのようなおかしなパスが生成されるエピソードは、体感1/6程度
- それ以外では、たまにオクルージョンでおかしなパスが生成されるケースがある
- パスのエラー率は、cleanの時と同程度かそれよりも少し多いぐらい

**move_can_pot**:
- canではないものをcanと誤認して、そこからパスが描写されているケースが非常に多い
- 体感では2/3ぐらいがcanを誤認している
- potは正しく認識されていそうなケースがほとんどだが、たまにpotも間違われている
- canがアームに持ち上げられている状態などではcanをcanと認識しないケースが目立つ
- cleanな環境で描写されたパスと比べて、パスの精度は非常に悪いと感じる

### 所感まとめ

- **Clean条件**: 全体的にパス品質は良好。オクルージョン発生時に一部誤認があるが、基本的なタスク方向・グリッパー動作は正しい
- **Cluttered条件**: タスクによって差が大きい
  - click_bell: 比較的良好だが、特定環境で失敗するエピソードがある
  - beat_block_hammer: オクルージョン時の誤認が激しく、タスク進行に伴い品質低下
  - move_can_pot: canの誤認が非常に多く（体感2/3）、パス精度は非常に悪い

### 評価時のパス可視化について (2025-01-06)

評価時のパスデータは保存されておらず、可視化は断念。ただし、cluttered訓練データと評価環境は同一のため、パス生成の傾向は概ね一致すると判断。訓練データの可視化結果で傾向把握は十分と結論。評価時に特有の問題（ポリシー失敗→異常姿勢→パス生成失敗の連鎖）が発生しうるが、これは根本原因ではなく結果であるため、訓練データのパス品質改善が優先事項。

---

## TODO List

### 直近のタスク（具体化済み）

| # | Task | Status | Details |
|---|------|--------|---------|
| 1 | `HAMSTER/results` のディレクトリ構造を確認 | [x] | clean/cluttered × 3タスク × 50エピソードの構成を把握 |
| 2 | 各エピソードからサンプルフレームを抽出（Frame 0, 16, 32, ...） | [x] | `analysis/path_quality/grids/` にグリッド画像を出力 |
| 3 | Frame 0 vs Frame N でパス品質を目視比較 | [x] | 上記「目視確認による所感」参照 |

### 今後のタスク（ざっくり）

| # | Task | Status | Details |
|---|------|--------|---------|
| 4 | 評価時のパスを可視化 | [x] | 断念。訓練データの可視化で傾向把握は十分と判断（2025-01-06） |
| 5 | 訓練 vs 評価のパス品質を比較・分析 | [-] | Step 2の結論によりスキップ |
| 6 | 改善方針を決定（Option A or B） | [x] | **Option A（GT + Noise）を採用**（2025-01-06） |
| 7 | 選択したOptionを実装 | [ ] | 下記「Option A 実装計画」参照 |
| 8 | 再訓練・評価 | [ ] | 後で具体化 |

---

## Option A 実装計画 (2025-01-06)

### 決定根拠

HAMSTER論文（Section 4.1.1, 4.2）に従い、**Ground Truth パスをproprioception投影で生成**する方式を採用。

論文からの引用:
> "The ground-truth 2D path is given by **proprioceptive projection using forward kinematics and camera parameters**" (Section 4.1.1)
> "During training, we use **oracle 2D paths constructed by proprioception projection**" (Section 4.2)

### RoboTwin 2.0 HDF5データ構造

```
episode{N}.hdf5
├── endpose/
│   ├── left_endpose    (T, 7)   # 3D EE位置 (xyz) + 姿勢 (quaternion)
│   ├── left_gripper    (T,)     # グリッパー状態
│   ├── right_endpose   (T, 7)
│   └── right_gripper   (T,)
├── observation/
│   ├── head_camera/
│   │   ├── intrinsic_cv  (T, 3, 3)  # カメラ内部パラメータ K
│   │   ├── extrinsic_cv  (T, 3, 4)  # カメラ外部パラメータ [R|t]
│   │   └── rgb           (T,)       # RGB画像パス
│   └── ...
```

**データパス**:
- `ManiFlow/third_party/RoboTwin2.0/dataset/dataset/{task}/aloha-agilex_{clean,cluttered}_50/data/episode{N}.hdf5`

### GTパス生成アルゴリズム

```python
def generate_gt_path(episode_data, frame_idx, camera='head_camera'):
    """
    フレームframe_idxにおけるGTパスを生成

    Returns:
        path: List[Tuple[float, float, int]]  # [(x, y, gripper_state), ...]
    """
    T = len(episode_data['endpose/left_endpose'])

    # 1. 現在フレームから終了までの3D EE軌跡を取得
    ee_positions_3d = episode_data['endpose/left_endpose'][frame_idx:T, :3]  # (T-frame_idx, 3)
    gripper_states = episode_data['endpose/left_gripper'][frame_idx:T]        # (T-frame_idx,)

    # 2. カメラパラメータ取得（frame_idxのものを使用）
    K = episode_data[f'observation/{camera}/intrinsic_cv'][frame_idx]  # (3, 3)
    Rt = episode_data[f'observation/{camera}/extrinsic_cv'][frame_idx]  # (3, 4)

    # 3. 3D → 2D投影
    #    P = K @ [R|t]
    #    p_2d = P @ [X, Y, Z, 1]^T
    P = K @ Rt  # (3, 4)
    ones = np.ones((len(ee_positions_3d), 1))
    ee_homogeneous = np.hstack([ee_positions_3d, ones])  # (N, 4)
    projected = (P @ ee_homogeneous.T).T  # (N, 3)

    # 同次座標 → 2D座標
    x_2d = projected[:, 0] / projected[:, 2]
    y_2d = projected[:, 1] / projected[:, 2]

    # 4. 正規化 (0-1)
    img_width, img_height = 320, 240  # RoboTwin 2.0 head_camera
    x_norm = x_2d / img_width
    y_norm = y_2d / img_height

    # 5. Ramer-Douglas-Peucker法で簡略化
    points = np.column_stack([x_norm, y_norm])
    simplified_indices = rdp_simplify(points, epsilon=0.02)  # 閾値は要調整

    # 6. グリッパー状態を付加
    path = []
    for idx in simplified_indices:
        gripper = 1 if gripper_states[idx] > 0.5 else 0  # OPEN=1, CLOSE=0
        path.append((x_norm[idx], y_norm[idx], gripper))

    # 7. グリッパー状態変化点を必ず含める
    path = ensure_gripper_changes(path, x_norm, y_norm, gripper_states)

    return path
```

### パス簡略化: Ramer-Douglas-Peucker法

```python
from rdp import rdp  # pip install rdp

def rdp_simplify(points, epsilon=0.02):
    """
    曲線を簡略化して点数を削減

    Args:
        points: (N, 2) array of (x, y) coordinates
        epsilon: 許容誤差（正規化座標系で0.02 = 画像の2%）

    Returns:
        simplified_indices: 残す点のインデックス
    """
    simplified = rdp(points, epsilon=epsilon, return_mask=True)
    return np.where(simplified)[0]
```

### 処理フロー

```
Step 1: HDF5からデータ読み込み
    ↓
Step 2: 各フレームで残り軌跡 (frame_idx → T) を取得
    ↓
Step 3: 3D EE位置をカメラパラメータで2D投影
    ↓
Step 4: Ramer-Douglas-Peucker法で簡略化
    ↓
Step 5: グリッパー状態変化点を付加
    ↓
Step 6: (Optional) ガウシアンノイズ追加
    ↓
Step 7: HAMSTER形式で保存
        - frames/frame_{idx:04d}.png
        - paths/frame_{idx:04d}.pkl
        - overlay_images/frame_{idx:04d}.png
```

### 出力ディレクトリ構造

```
HAMSTER/results/gt_paths_{clean,cluttered}/
├── {task}/
│   ├── episode_00/
│   │   ├── frames/
│   │   │   ├── frame_0000.png
│   │   │   ├── frame_0001.png
│   │   │   └── ...
│   │   ├── paths/
│   │   │   ├── frame_0000.pkl  # [(x, y, gripper), ...]
│   │   │   ├── frame_0001.pkl
│   │   │   └── ...
│   │   └── overlay_images/
│   │       ├── frame_0000.png
│   │       ├── frame_0001.png
│   │       └── ...
│   ├── episode_01/
│   └── ...
```

### 実装タスク

| # | Task | Status | Details |
|---|------|--------|---------|
| A1 | GTパス生成スクリプト作成 | [x] | `HAMSTER/tests/gt_path_generation/generate_gt_paths.py` |
| A2 | 3D→2D投影の検証 | [x] | グリッパー先端位置で正確に投影されることを確認 |
| A3 | 全タスク対応 | [x] | 3タスクすべてで「グラスプ以降のみ」のパス生成ロジックを統一 |
| A4 | オーバーレイ画像生成 | [x] | HAMSTER式描画（`overlay_utils.py`）を使用 |
| A5 | **全エピソードGTパス生成** | [x] | **完了 (2025-01-07)**: clean/cluttered × 3タスク × 50エピソード = 300エピソード |
| A6 | Zarr変換スクリプト修正 | [ ] | GTパス用の変換対応 |
| A7 | ManiFlow再トレーニング | [ ] | 6条件 × 3タスク |
| A8 | 評価・結果比較 | [ ] | VILAパス vs GTパスの比較 |

### 実装詳細 (2025-01-06 完了)

#### 1. グリッパー先端位置の計算

EE位置（endpose）はグリッパーの付け根位置であり、実際に物体を掴む先端位置とはオフセットがある。

**オフセット計算根拠** (2025-01-07 更新):
- Link6 → フィンガー付け根 (URDFのfl_joint7/fl_joint8): **84.57mm**
- フィンガーメッシュの先端 (STL max X座標): **71mm**
- 実際の接触点は先端から約10mm手前: 71 - 10 = **61mm**
- 合計: 84.57 + 61 ≈ **146mm**

```python
from scipy.spatial.transform import Rotation as R

# グリッパー先端オフセット (2025-01-07 更新: 85mm → 146mm)
GRIPPER_TIP_OFFSET = np.array([-0.146, 0.0, 0.0])  # ローカル座標系

def compute_gripper_tip_positions(ee_positions, ee_quaternions):
    """EE位置と姿勢からグリッパー先端位置を計算"""
    tip_positions = np.zeros_like(ee_positions)
    for i in range(len(ee_positions)):
        rot = R.from_quat(ee_quaternions[i])
        rot_matrix = rot.as_matrix()
        world_offset = rot_matrix @ GRIPPER_TIP_OFFSET
        tip_positions[i] = ee_positions[i] + world_offset
    return tip_positions
```

#### 2. アクティブアーム自動検出

RoboTwin 2.0はバイマニュアル（双腕）ロボットで、エピソードごとに左右どちらのアームがタスクを実行するかが異なる。
エピソード内の移動量を比較して自動検出する。

```python
def detect_active_arm(left_endpose, right_endpose):
    left_movement = np.sum(np.abs(np.diff(left_endpose[:, :3], axis=0)))
    right_movement = np.sum(np.abs(np.diff(right_endpose[:, :3], axis=0)))
    return "left" if left_movement > right_movement else "right"
```

#### 3. タスク固有パス生成ロジック（全タスク共通化）

**全タスク共通**（beat_block_hammer, click_bell, move_can_pot）:
- グリッパーがオブジェクトを掴む前のフレーム: 掴む位置（first_grasp_frame）→ 最終位置のパス
- 掴んだ後のフレーム: 現在位置 → 最終位置のパス
- オブジェクトを取りに行く動作はパスに含めない（掴んでからの動作のみ）

```python
def generate_gt_path_from_grasp(tip_positions, gripper_states, ...):
    """全タスク共通: グラスプ以降のみを表示するGTパス生成"""
    first_grasp_frame = find_first_grasp_frame(gripper_states)  # OPEN→CLOSEの最初のフレーム

    if frame_idx < first_grasp_frame:
        start_frame = first_grasp_frame  # 掴む前: grasp位置からのパス
    else:
        start_frame = frame_idx  # 掴んだ後: 現在位置からのパス

    # start_frame → 最終フレーム の軌跡を投影・簡略化
```

#### 4. 画像サイズ

RoboTwin 2.0のhead_cameraは**320x240**。

#### 5. HAMSTER式描画

既存の`HAMSTER/tests/training_data/overlay_utils.py`の`draw_path_on_image_hamster_style()`を使用:
- 線の色: jet colormap（青→シアン→緑→黄→赤）
- パス補間: 100分割して滑らかに描画
- グリッパーマーカー: 状態変化点のみ、輪郭のみ（Open=青、Close=赤）

### ノイズ追加

評価時はVILAパスを使用するため、訓練時にノイズを追加してVILA誤差への頑健性を獲得。

**HAMSTER論文の設定** (Section 4.2):
> "During training, we add Gaussian noise σ = 0.01 to oracle path points"

```python
def add_gaussian_noise(path, sigma=0.01, gripper_flip_prob=0.0):
    """
    Args:
        sigma: 座標ノイズの標準偏差（正規化座標系）- HAMSTER論文準拠で0.01
        gripper_flip_prob: グリッパー状態反転確率（デフォルト0）
    """
    noisy_path = []
    for x, y, g in path:
        x_noisy = x + np.random.normal(0, sigma)
        y_noisy = y + np.random.normal(0, sigma)
        g_noisy = 1 - g if np.random.random() < gripper_flip_prob else g
        noisy_path.append((
            np.clip(x_noisy, 0, 1),
            np.clip(y_noisy, 0, 1),
            g_noisy
        ))
    return noisy_path
```

**生成済みデータ (2025-01-07)**:
- sigma=0.01のガウシアンノイズを全GTパスに適用
- 出力先: `HAMSTER/results/gt_paths_{clean,cluttered}/`

---

## GTパス生成結果サマリー (2025-01-07)

### 生成設定

| 設定項目 | 値 |
|---------|-----|
| グリッパー先端オフセット | 146mm |
| RDP epsilon | 0.02 |
| ガウシアンノイズ sigma | 0.01 |

### 生成結果

| 環境 | タスク | エピソード数 | 成功率 | 平均パスポイント数 |
|------|--------|------------|--------|-------------------|
| clean | beat_block_hammer | 50 | 100% | 4.2 |
| clean | click_bell | 50 | 100% | 2.4 |
| clean | move_can_pot | 50 | 100% | 4.6 |
| cluttered | beat_block_hammer | 50 | 100% | 4.1 |
| cluttered | click_bell | 50 | 100% | 2.4 |
| cluttered | move_can_pot | 50 | 100% | 4.5 |
| **合計** | **6タスク** | **300** | **100%** | - |

### 出力ディレクトリ

```
HAMSTER/results/
├── gt_paths_clean/
│   ├── beat_block_hammer/episode_{00-49}/
│   ├── click_bell/episode_{00-49}/
│   └── move_can_pot/episode_{00-49}/
└── gt_paths_cluttered/
    ├── beat_block_hammer/episode_{00-49}/
    ├── click_bell/episode_{00-49}/
    └── move_can_pot/episode_{00-49}/
```

---

## References

- HAMSTER: [arXiv:2502.05485](https://arxiv.org/abs/2502.05485)
- PEEK: (論文リンクTBD)
- Current implementation: `ManiFlow/ManiFlow/maniflow/dataset/robotwin2_overlay_zarr_dataset.py`

