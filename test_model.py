import os
import torch
import cv2

# モデルのパスを設定
model_path = 'yolov5/runs/train/exp5/weights/best.pt'  # トレーニング時の保存先

# トレーニング済みモデルをロード
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# 入力画像フォルダのパス
input_folder_path = 'test_image'  # 処理する画像が格納されているフォルダ
output_folder_path = 'output_images'  # 処理結果を保存するフォルダ

# 出力フォルダを作成（存在しない場合）
os.makedirs(output_folder_path, exist_ok=True)

# 画像ファイルを取得
image_files = [f for f in os.listdir(input_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(image_files)} images to process...")

# クラスIDに対応するラベルを定義
class_labels = {0: "Face", 1: "Hand"}

# 画像ごとに処理
for idx, image_file in enumerate(image_files):
    # 入力画像のパスを取得
    input_image_path = os.path.join(input_folder_path, image_file)
    output_image_path = os.path.join(output_folder_path, image_file)

    # 画像を読み込む
    frame = cv2.imread(input_image_path)

    # BGR画像をRGBに変換 (YOLOモデルの入力形式に合わせる)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 推論を実行
    results = model(rgb_frame)

    # 検出結果を取得
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

    # 閾値を適用してフィルタリング
    conf_threshold = 0.001  # 閾値を設定
    detections = [det for det in detections if det[4] >= conf_threshold]

    # クラスごとにフィルタリング
    face_detections = [det for det in detections if int(det[5]) == 0]  # クラスID 0: Face
    hand_detections = [det for det in detections if int(det[5]) == 1]  # クラスID 1: Hand

    # スコアでソートして上位を取得
    face_detections = sorted(face_detections, key=lambda x: x[4], reverse=True)[:1]  # 上位1つ
    hand_detections = sorted(hand_detections, key=lambda x: x[4], reverse=True)[:2]  # 上位2つ

    # 描画用に統合
    top_detections = face_detections + hand_detections

    # 検出結果を描画
    for det in top_detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_id = int(cls)
        label = class_labels.get(class_id, f"Class {class_id}")  # ラベルを取得（デフォルトはクラスID）
        label_text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 処理結果を保存
    cv2.imwrite(output_image_path, frame)

    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(image_files)} images...")

print(f"All images processed and saved in {output_folder_path}")