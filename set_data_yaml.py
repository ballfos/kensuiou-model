data_yaml = """
nc: 2
names:
  - face  # クラスID 0: 顔
  - hand  # クラスID 1: 手

train: ../annotated_frames/images/train  # トレーニング画像のパス
val: ../annotated_frames/images/val     # 検証用画像のパス
"""

with open("yolov5/data.yaml", "w") as f:
    f.write(data_yaml)