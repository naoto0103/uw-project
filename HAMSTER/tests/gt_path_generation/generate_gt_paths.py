#!/usr/bin/env python3
"""
Ground Truth Path Generation Script

RoboTwin 2.0のHDF5データからGround Truthパスを生成する。
HAMSTERの論文に従い、proprioception（3D EE位置）をカメラパラメータで2D投影し、
Ramer-Douglas-Peucker法で簡略化する。

Usage:
    python generate_gt_paths.py --task beat_block_hammer --env clean --episodes 50
    python generate_gt_paths.py --task all --env all --episodes 50
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import json

import numpy as np
import h5py
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Ramer-Douglas-Peucker法
try:
    from rdp import rdp
except ImportError:
    print("rdp not found. Install with: pip install rdp")
    raise

# HAMSTER式のオーバーレイ描画をインポート
sys.path.insert(0, str(Path(__file__).parent.parent / "training_data"))
from overlay_utils import draw_path_on_image_hamster_style


# =============================================================================
# Configuration
# =============================================================================

TASKS = ["beat_block_hammer", "click_bell", "move_can_pot"]
ENVS = ["clean", "cluttered"]

# データパス
ROBOTWIN_DATASET_BASE = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/ManiFlow/third_party/RoboTwin2.0/dataset/dataset")
OUTPUT_BASE = Path("/gscratch/scrubbed/naoto03/projects/HAMSTER-ManiFlow-Integration/HAMSTER/results")

# RDP簡略化パラメータ
RDP_EPSILON = 0.02  # 正規化座標系での許容誤差（画像の2%）

# 画像サイズ（RoboTwin 2.0のデフォルト: 320x240）
IMG_WIDTH = 320
IMG_HEIGHT = 240

# グリッパー状態の閾値
GRIPPER_OPEN_THRESHOLD = 0.5  # これ以上ならOPEN

# グリッパー先端オフセット
# 計算根拠:
#   - Link6 → フィンガー付け根 (URDF): 84.57mm
#   - フィンガーメッシュの先端 (max X): 71mm
#   - 実際の接触点は先端から約10mm手前: 71 - 10 = 61mm
#   - 合計: 84.57 + 61 ≈ 146mm
# 注: URDFではX+方向がアーム側、X-方向がグリッパー先端側
GRIPPER_TIP_OFFSET = np.array([-0.146, 0.0, 0.0])


# =============================================================================
# Core Functions
# =============================================================================

def compute_gripper_tip_positions(
    ee_positions: np.ndarray,
    ee_quaternions: np.ndarray
) -> np.ndarray:
    """
    EE位置と姿勢からグリッパー先端位置を計算

    Args:
        ee_positions: (T, 3) EE位置 (xyz)
        ee_quaternions: (T, 4) EE姿勢 (quaternion: x, y, z, w)

    Returns:
        tip_positions: (T, 3) グリッパー先端位置
    """
    T = len(ee_positions)
    tip_positions = np.zeros_like(ee_positions)

    for i in range(T):
        pos = ee_positions[i]
        quat = ee_quaternions[i]

        # Quaternionから回転行列を取得
        rot = R.from_quat(quat)
        rot_matrix = rot.as_matrix()

        # ローカルX軸方向のオフセットをワールド座標に変換
        world_offset = rot_matrix @ GRIPPER_TIP_OFFSET
        tip_positions[i] = pos + world_offset

    return tip_positions


def project_3d_to_2d(
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray
) -> np.ndarray:
    """
    3Dポイントを2D画像座標に投影

    Args:
        points_3d: (N, 3) 3D座標
        intrinsic: (3, 3) カメラ内部パラメータ K
        extrinsic: (3, 4) カメラ外部パラメータ [R|t]

    Returns:
        points_2d: (N, 2) 2D画像座標
    """
    N = len(points_3d)

    # 同次座標に変換
    ones = np.ones((N, 1))
    points_homo = np.hstack([points_3d, ones])  # (N, 4)

    # 投影行列 P = K @ [R|t]
    P = intrinsic @ extrinsic  # (3, 4)

    # 投影
    projected = (P @ points_homo.T).T  # (N, 3)

    # 同次座標から2D座標へ
    x_2d = projected[:, 0] / projected[:, 2]
    y_2d = projected[:, 1] / projected[:, 2]

    return np.column_stack([x_2d, y_2d])


def rdp_simplify(points: np.ndarray, epsilon: float = RDP_EPSILON) -> np.ndarray:
    """
    Ramer-Douglas-Peucker法で曲線を簡略化

    Args:
        points: (N, 2) 座標配列
        epsilon: 許容誤差

    Returns:
        mask: (N,) bool配列、残す点がTrue
    """
    if len(points) <= 2:
        return np.ones(len(points), dtype=bool)

    # rdpライブラリはマスクを返すオプションがある
    mask = rdp(points, epsilon=epsilon, return_mask=True)
    return mask


def find_gripper_change_indices(gripper_states: np.ndarray) -> List[int]:
    """
    グリッパー状態が変化するインデックスを見つける

    Args:
        gripper_states: (N,) グリッパー状態配列

    Returns:
        change_indices: 状態変化点のインデックスリスト
    """
    binary_states = (gripper_states > GRIPPER_OPEN_THRESHOLD).astype(int)
    changes = np.diff(binary_states)
    change_indices = np.where(changes != 0)[0] + 1  # 変化後のインデックス
    return change_indices.tolist()


def generate_gt_path_for_frame(
    ee_positions_3d: np.ndarray,
    gripper_states: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    frame_idx: int,
    epsilon: float = RDP_EPSILON
) -> List[Tuple[float, float, int]]:
    """
    特定フレームのGTパスを生成

    Args:
        ee_positions_3d: (T, 3) 全フレームの3D EE位置
        gripper_states: (T,) 全フレームのグリッパー状態
        intrinsic: (3, 3) カメラ内部パラメータ
        extrinsic: (3, 4) カメラ外部パラメータ
        frame_idx: 現在のフレームインデックス
        epsilon: RDP簡略化の許容誤差

    Returns:
        path: [(x, y, gripper_state), ...] 正規化座標
    """
    T = len(ee_positions_3d)

    # 現在フレームから終了までの軌跡
    remaining_ee = ee_positions_3d[frame_idx:T]
    remaining_gripper = gripper_states[frame_idx:T]

    if len(remaining_ee) == 0:
        return []

    # 3D → 2D投影
    points_2d = project_3d_to_2d(remaining_ee, intrinsic, extrinsic)

    # 正規化 (0-1)
    x_norm = points_2d[:, 0] / IMG_WIDTH
    y_norm = points_2d[:, 1] / IMG_HEIGHT

    # クリッピング（画像外に出た場合）
    x_norm = np.clip(x_norm, 0, 1)
    y_norm = np.clip(y_norm, 0, 1)

    points_norm = np.column_stack([x_norm, y_norm])

    # RDP簡略化
    rdp_mask = rdp_simplify(points_norm, epsilon=epsilon)

    # グリッパー状態変化点を必ず含める
    gripper_change_indices = find_gripper_change_indices(remaining_gripper)
    for idx in gripper_change_indices:
        if idx < len(rdp_mask):
            rdp_mask[idx] = True

    # 始点と終点は必ず含める
    rdp_mask[0] = True
    rdp_mask[-1] = True

    # パス構築
    path = []
    selected_indices = np.where(rdp_mask)[0]

    for idx in selected_indices:
        x = float(x_norm[idx])
        y = float(y_norm[idx])
        gripper = 1 if remaining_gripper[idx] > GRIPPER_OPEN_THRESHOLD else 0
        path.append((x, y, gripper))

    return path


def find_first_grasp_frame(gripper_states: np.ndarray) -> Optional[int]:
    """
    最初のグラスプ（OPEN→CLOSE）フレームを見つける

    Args:
        gripper_states: (T,) グリッパー状態配列

    Returns:
        グラスプフレームのインデックス、なければNone
    """
    binary = (gripper_states > GRIPPER_OPEN_THRESHOLD).astype(int)  # 1=open, 0=close
    for i in range(1, len(binary)):
        if binary[i-1] == 1 and binary[i] == 0:  # OPEN→CLOSE
            return i
    return None


def generate_gt_path_from_grasp(
    ee_positions_3d: np.ndarray,
    gripper_states: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    frame_idx: int,
    first_grasp_frame: int,
    epsilon: float = RDP_EPSILON
) -> List[Tuple[float, float, int]]:
    """
    グラスプ以降のみを表示するGTパス生成（全タスク共通）

    ロジック:
    - Frame 0〜(grasp-1): grasp位置 → 最終位置 のパス
    - Frame grasp以降: 現在位置 → 最終位置 のパス

    Args:
        ee_positions_3d: (T, 3) 全フレームの3D EE位置
        gripper_states: (T,) 全フレームのグリッパー状態
        intrinsic: (3, 3) カメラ内部パラメータ
        extrinsic: (3, 4) カメラ外部パラメータ
        frame_idx: 現在のフレームインデックス
        first_grasp_frame: オブジェクトを掴むフレーム
        epsilon: RDP簡略化の許容誤差

    Returns:
        path: [(x, y, gripper_state), ...] 正規化座標
    """
    T = len(ee_positions_3d)

    # パスの開始フレームを決定
    if frame_idx < first_grasp_frame:
        # 掴む前: grasp位置からのパス
        start_frame = first_grasp_frame
    else:
        # 掴んだ後: 現在位置からのパス
        start_frame = frame_idx

    # 開始フレームから終了までの軌跡
    remaining_ee = ee_positions_3d[start_frame:T]
    remaining_gripper = gripper_states[start_frame:T]

    if len(remaining_ee) == 0:
        return []

    # 3D → 2D投影
    points_2d = project_3d_to_2d(remaining_ee, intrinsic, extrinsic)

    # 正規化 (0-1)
    x_norm = points_2d[:, 0] / IMG_WIDTH
    y_norm = points_2d[:, 1] / IMG_HEIGHT

    # クリッピング（画像外に出た場合）
    x_norm = np.clip(x_norm, 0, 1)
    y_norm = np.clip(y_norm, 0, 1)

    points_norm = np.column_stack([x_norm, y_norm])

    # RDP簡略化
    rdp_mask = rdp_simplify(points_norm, epsilon=epsilon)

    # グリッパー状態変化点を必ず含める
    gripper_change_indices = find_gripper_change_indices(remaining_gripper)
    for idx in gripper_change_indices:
        if idx < len(rdp_mask):
            rdp_mask[idx] = True

    # 始点と終点は必ず含める
    rdp_mask[0] = True
    rdp_mask[-1] = True

    # パス構築
    path = []
    selected_indices = np.where(rdp_mask)[0]

    for idx in selected_indices:
        x = float(x_norm[idx])
        y = float(y_norm[idx])
        gripper = 1 if remaining_gripper[idx] > GRIPPER_OPEN_THRESHOLD else 0
        path.append((x, y, gripper))

    return path


# =============================================================================
# Data Loading
# =============================================================================

def detect_active_arm(left_endpose: np.ndarray, right_endpose: np.ndarray) -> str:
    """
    エピソード内でどちらのアームがアクティブかを検出

    Args:
        left_endpose: (T, 7) 左アームのEEポーズ
        right_endpose: (T, 7) 右アームのEEポーズ

    Returns:
        "left" or "right"
    """
    left_movement = np.sum(np.abs(np.diff(left_endpose[:, :3], axis=0)))
    right_movement = np.sum(np.abs(np.diff(right_endpose[:, :3], axis=0)))

    return "left" if left_movement > right_movement else "right"


def load_episode_data(hdf5_path: Path) -> dict:
    """
    HDF5ファイルからエピソードデータを読み込む

    Returns:
        dict with keys:
            - left_endpose: (T, 7)
            - left_gripper: (T,)
            - right_endpose: (T, 7)
            - right_gripper: (T,)
            - head_camera_intrinsic: (T, 3, 3)
            - head_camera_extrinsic: (T, 3, 4)
            - head_camera_rgb_bytes: list of bytes (JPEG data)
            - active_arm: "left" or "right"
    """
    data = {}

    with h5py.File(hdf5_path, 'r') as f:
        # End effector poses
        data['left_endpose'] = f['endpose/left_endpose'][:]
        data['left_gripper'] = f['endpose/left_gripper'][:]
        data['right_endpose'] = f['endpose/right_endpose'][:]
        data['right_gripper'] = f['endpose/right_gripper'][:]

        # Camera parameters
        data['head_camera_intrinsic'] = f['observation/head_camera/intrinsic_cv'][:]
        data['head_camera_extrinsic'] = f['observation/head_camera/extrinsic_cv'][:]

        # RGB images (stored as JPEG bytes directly in HDF5)
        rgb_data = f['observation/head_camera/rgb'][:]
        data['head_camera_rgb_bytes'] = [bytes(p) for p in rgb_data]

    # アクティブアームを自動検出
    data['active_arm'] = detect_active_arm(data['left_endpose'], data['right_endpose'])

    return data


def decode_jpeg_bytes(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    """
    JPEGバイトデータをBGR画像にデコード

    Args:
        jpeg_bytes: JPEG画像のバイナリデータ

    Returns:
        BGR画像 or None
    """
    if not jpeg_bytes:
        return None

    # バイト列からnumpy配列に変換してデコード
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


# =============================================================================
# Main Processing
# =============================================================================

def process_episode(
    task: str,
    env: str,
    episode_idx: int,
    output_base: Path,
    add_noise: bool = False,
    noise_sigma: float = 0.02,
    save_overlay: bool = True
) -> dict:
    """
    1エピソードを処理してGTパスを生成

    Returns:
        stats: 統計情報
    """
    # データパス
    env_suffix = f"aloha-agilex_{env}_50"
    dataset_dir = ROBOTWIN_DATASET_BASE / task / env_suffix
    hdf5_path = dataset_dir / "data" / f"episode{episode_idx}.hdf5"

    if not hdf5_path.exists():
        return {"status": "not_found", "episode": episode_idx}

    # 出力ディレクトリ
    output_dir = output_base / f"gt_paths_{env}" / task / f"episode_{episode_idx:02d}"
    frames_dir = output_dir / "frames"
    paths_dir = output_dir / "paths"
    overlay_dir = output_dir / "overlay_images"

    frames_dir.mkdir(parents=True, exist_ok=True)
    paths_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    data = load_episode_data(hdf5_path)
    T = len(data['left_endpose'])

    # アクティブアームを自動検出して使用
    active_arm = data['active_arm']
    if active_arm == "left":
        ee_positions = data['left_endpose'][:, :3]  # (T, 3) - xyz only
        ee_quaternions = data['left_endpose'][:, 3:7]  # (T, 4) - quaternion
        gripper_states = data['left_gripper']  # (T,)
    else:
        ee_positions = data['right_endpose'][:, :3]  # (T, 3) - xyz only
        ee_quaternions = data['right_endpose'][:, 3:7]  # (T, 4) - quaternion
        gripper_states = data['right_gripper']  # (T,)

    # グリッパー先端位置を計算（EE位置 + 姿勢に基づくオフセット）
    tip_positions = compute_gripper_tip_positions(ee_positions, ee_quaternions)

    # 全タスク共通: 最初のグラスプフレームを検出
    first_grasp_frame = find_first_grasp_frame(gripper_states)
    if first_grasp_frame is None:
        first_grasp_frame = 0  # フォールバック（グラスプがない場合は最初から）

    stats = {
        "status": "success",
        "episode": episode_idx,
        "total_frames": T,
        "active_arm": active_arm,
        "first_grasp_frame": first_grasp_frame,
        "paths_generated": 0,
        "avg_path_points": 0,
    }

    path_points_list = []

    for frame_idx in range(T):
        # カメラパラメータ（フレームごとに異なる可能性あり）
        intrinsic = data['head_camera_intrinsic'][frame_idx]
        extrinsic = data['head_camera_extrinsic'][frame_idx]

        # GTパス生成（全タスク共通: グラスプ以降のみ）- グリッパー先端位置を使用
        path = generate_gt_path_from_grasp(
            tip_positions,
            gripper_states,
            intrinsic,
            extrinsic,
            frame_idx,
            first_grasp_frame,
            epsilon=RDP_EPSILON
        )

        # ノイズ追加（オプション）
        if add_noise and len(path) > 0:
            path = add_gaussian_noise(path, sigma=noise_sigma)

        # パス保存
        path_file = paths_dir / f"frame_{frame_idx:04d}.pkl"
        with open(path_file, 'wb') as f:
            pickle.dump(path, f)

        path_points_list.append(len(path))
        stats["paths_generated"] += 1

        # 画像とオーバーレイ
        if save_overlay or frame_idx == 0:  # 少なくともframe 0は保存
            jpeg_bytes = data['head_camera_rgb_bytes'][frame_idx]
            image = decode_jpeg_bytes(jpeg_bytes)

            if image is not None:
                # フレーム保存
                frame_file = frames_dir / f"frame_{frame_idx:04d}.png"
                cv2.imwrite(str(frame_file), image)

                # オーバーレイ保存（HAMSTER式描画）
                if save_overlay and len(path) > 0:
                    overlay = draw_path_on_image_hamster_style(image, path, num_subdivisions=100)
                    overlay_file = overlay_dir / f"frame_{frame_idx:04d}.png"
                    cv2.imwrite(str(overlay_file), overlay)

    if path_points_list:
        stats["avg_path_points"] = np.mean(path_points_list)

    return stats


def add_gaussian_noise(
    path: List[Tuple[float, float, int]],
    sigma: float = 0.01,
    gripper_flip_prob: float = 0.0
) -> List[Tuple[float, float, int]]:
    """
    パスにガウシアンノイズを追加

    Args:
        path: 元のパス
        sigma: 座標ノイズの標準偏差
        gripper_flip_prob: グリッパー状態反転確率

    Returns:
        ノイズ追加後のパス
    """
    noisy_path = []
    for x, y, g in path:
        x_noisy = np.clip(x + np.random.normal(0, sigma), 0, 1)
        y_noisy = np.clip(y + np.random.normal(0, sigma), 0, 1)

        if gripper_flip_prob > 0 and np.random.random() < gripper_flip_prob:
            g = 1 - g

        noisy_path.append((float(x_noisy), float(y_noisy), int(g)))

    return noisy_path


def main():
    parser = argparse.ArgumentParser(description="Generate Ground Truth paths from RoboTwin 2.0 data")
    parser.add_argument("--task", type=str, default="all",
                        help="Task name or 'all'")
    parser.add_argument("--env", type=str, default="all",
                        help="Environment: clean, cluttered, or 'all'")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes to process")
    parser.add_argument("--epsilon", type=float, default=RDP_EPSILON,
                        help="RDP simplification epsilon")
    parser.add_argument("--add-noise", action="store_true",
                        help="Add Gaussian noise to paths")
    parser.add_argument("--noise-sigma", type=float, default=0.02,
                        help="Noise standard deviation")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip overlay image generation (faster)")
    parser.add_argument("--output-base", type=str, default=str(OUTPUT_BASE),
                        help="Output base directory")

    args = parser.parse_args()

    # RDP epsilon（ローカル変数として使用）
    rdp_eps = args.epsilon

    # タスクと環境のリスト
    tasks = TASKS if args.task == "all" else [args.task]
    envs = ENVS if args.env == "all" else [args.env]

    output_base = Path(args.output_base)

    print("=" * 60)
    print("Ground Truth Path Generation")
    print("=" * 60)
    print(f"Tasks: {tasks}")
    print(f"Environments: {envs}")
    print(f"Episodes: {args.episodes}")
    print(f"RDP Epsilon: {args.epsilon}")
    print(f"Add Noise: {args.add_noise}")
    print(f"Output: {output_base}")
    print("=" * 60)

    all_stats = {}

    for env in envs:
        for task in tasks:
            print(f"\nProcessing: {env}/{task}")
            task_stats = []

            for ep_idx in tqdm(range(args.episodes), desc=f"{task}"):
                stats = process_episode(
                    task=task,
                    env=env,
                    episode_idx=ep_idx,
                    output_base=output_base,
                    add_noise=args.add_noise,
                    noise_sigma=args.noise_sigma,
                    save_overlay=not args.no_overlay
                )
                task_stats.append(stats)

            # 統計まとめ
            successful = [s for s in task_stats if s["status"] == "success"]
            print(f"  Completed: {len(successful)}/{args.episodes}")
            if successful:
                avg_points = np.mean([s["avg_path_points"] for s in successful])
                print(f"  Avg path points: {avg_points:.1f}")

            all_stats[f"{env}/{task}"] = task_stats

    # 統計保存
    stats_file = output_base / "gt_path_generation_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nStats saved to: {stats_file}")
    print("Done!")


if __name__ == "__main__":
    main()
