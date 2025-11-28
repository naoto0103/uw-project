#!/usr/bin/env python3
"""
HAMSTERサーバーAPIをテストするシンプルなクライアント
"""
import base64
import cv2
import numpy as np
from openai import OpenAI
import re
import os

# サーバーIP設定
SERVER_IP_FILE = "./ip_eth0.txt"
try:
    with open(SERVER_IP_FILE, "r") as f:
        SERVER_IP = f.read().strip()
except Exception:
    SERVER_IP = "127.0.0.1"

print(f"サーバーに接続: {SERVER_IP}:8000")

# グリッパー状態の定義
GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1

def process_answer(input_str):
    """
    モデルの出力から経路点を抽出
    例: [(0.25, 0.32), (0.32, 0.17), <action>Open Gripper</action>, ...]
    戻り値: [[x, y, gripper_state], ...]
    """
    pattern = r'\(([0-9.]+),\s*([0-9.]+)\)|<action>(.*?)</action>'
    matches = re.findall(pattern, input_str)

    processed_points = []
    action_flag = GRIPPER_CLOSE  # デフォルトは閉じた状態

    for match in matches:
        x, y, action = match
        if action:  # アクション指示の場合
            action_lower = action.lower()
            if 'close' in action_lower:
                action_flag = GRIPPER_CLOSE
                if processed_points:
                    processed_points[-1][-1] = action_flag
            elif 'open' in action_lower:
                action_flag = GRIPPER_OPEN
                if processed_points:
                    processed_points[-1][-1] = action_flag
        else:  # 座標の場合
            x, y = float(x), float(y)
            # 特殊な座標値（1000.0, 1001.0）の処理
            if x == y == 1000.0:
                action_flag = GRIPPER_CLOSE
                if processed_points:
                    processed_points[-1][-1] = action_flag
                continue
            elif x == y == 1001.0:
                action_flag = GRIPPER_OPEN
                if processed_points:
                    processed_points[-1][-1] = action_flag
                continue
            processed_points.append([x, y, action_flag])

    return processed_points

def draw_path_on_image(image, points):
    """
    画像上に経路を描画
    points: [[x, y, gripper_state], ...]（正規化座標 0-1）
    """
    height, width = image.shape[:2]
    output_image = image.copy()

    # スケールファクターの計算
    scale_factor = max(min(width, height) / 512.0, 1)
    circle_radius = int(7 * scale_factor)
    line_thickness = max(1, int(3 * scale_factor))

    # 正規化座標をピクセル座標に変換
    pixel_points = []
    gripper_status = []
    for point in points:
        x = int(point[0] * width)
        y = int(point[1] * height)
        action = int(point[2])
        pixel_points.append((x, y))
        gripper_status.append(action)

    # 経路線を描画
    for i in range(len(pixel_points) - 1):
        color = (0, 255, 0)  # 緑色
        cv2.line(output_image, pixel_points[i], pixel_points[i+1], color, line_thickness)

    # グリッパー状態が変化する点にマーカーを描画
    for idx, (x, y) in enumerate(pixel_points):
        if idx == 0 or gripper_status[idx] != gripper_status[idx - 1]:
            # 赤=閉じる、青=開く
            circle_color = (0, 0, 255) if gripper_status[idx] == GRIPPER_CLOSE else (255, 0, 0)
            cv2.circle(output_image, (x, y), circle_radius, circle_color, -1)

    # 開始点を特別にマーク（黄色の円）
    if pixel_points:
        cv2.circle(output_image, pixel_points[0], circle_radius + 3, (0, 255, 255), 2)

    return output_image

def test_api(image_path, quest, output_path=None):
    """
    HAMSTER APIをテストして経路を生成
    """
    print(f"\n{'='*60}")
    print(f"画像: {image_path}")
    print(f"タスク: {quest}")
    print(f"{'='*60}")

    # 画像を読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: 画像を読み込めません: {image_path}")
        return

    print(f"画像サイズ: {image.shape}")

    # 画像をbase64エンコード
    _, encoded_image_array = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(encoded_image_array.tobytes()).decode('utf-8')

    # APIリクエスト送信
    print(f"APIリクエストを送信中...")
    try:
        client = OpenAI(base_url=f"http://{SERVER_IP}:8000", api_key="fake-key")
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                        {"type": "text", "text":
                            f"\nIn the image, please execute the command described in <quest>{quest}</quest>.\n"
                            "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\n"
                            "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
                            "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close Gripper</action>, ...]</ans>\n"
                            "The tuple denotes point x and y location of the end effector in the image. The action tags indicate gripper actions.\n"
                            "Coordinates should be floats between 0 and 1, representing relative positions.\n"
                            "Remember to provide points between <ans> and </ans> tags and think step by step."
                        },
                    ],
                }
            ],
            max_tokens=128,
            model="HAMSTER_dev",
            extra_body={"num_beams": 1, "use_cache": False, "temperature": 0.0, "top_p": 0.95},
        )
        print("✅ APIレスポンスを受信")
    except Exception as e:
        print(f"❌ APIリクエストエラー: {e}")
        return

    # レスポンスを解析
    response_text = response.choices[0].message.content[0]['text']
    print(f"\nモデルのレスポンス:\n{response_text}")

    # 経路点を抽出
    try:
        ans_match = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL)
        if not ans_match:
            print("❌ エラー: <ans>タグが見つかりません")
            return

        response_text_strip = ans_match.group(1)
        points = process_answer(response_text_strip)

        print(f"\n✅ 経路点を抽出: {len(points)}点")
        print(f"最初の5点: {points[:5]}")
        if len(points) > 5:
            print(f"最後の5点: {points[-5:]}")

        # 経路を画像上に描画
        output_image = draw_path_on_image(image, points)

        # 出力ファイル名を決定
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"output_{base_name}_result.jpg"

        cv2.imwrite(output_path, output_image)
        print(f"\n✅ 結果画像を保存: {output_path}")

    except Exception as e:
        print(f"❌ 経路の解析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # テストケース
    test_cases = [
        ("examples/non_prehensile.jpg", "open the top drawer"),
        ("examples/spatial_world_knowledge.jpg", "Have the middle block on Jensen Huang"),
        # ocr_reasoning.jpgは大きいので最後にテスト
        # ("examples/ocr_reasoning.jpg", "Move the S to the plate the arrow is pointing at"),
    ]

    for image_path, quest in test_cases:
        test_api(image_path, quest)
        print("\n")
