# 必要なモジュールを読み込む
# Flask関連
from flask import Flask, render_template, request, redirect, url_for, abort


from datetime import datetime

import cv2
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image #, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device('cpu')
print(device)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        # アプロードされたファイルをいったん保存する
        f = request.files["file"]
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
        f.save(filepath)
        # 画像ファイルを読み込む
        image = Image.open(filepath).convert("RGB")
        
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            predictions = model(
                image.to(device)
            )

            _predictions = []
            for pred in predictions:
                mask = np.argmax(pred['scores'])
                _predictions.append(
                    {
                        'boxes': pred['boxes'][mask],
                        'labels': pred['labels'][mask],
                        'scores': pred['scores'][mask],
                    }
                )
            predictions = _predictions

        print(predictions)

        print(COCO_INSTANCE_CATEGORY_NAMES[predictions[0]['labels'].item()])

        # 予測を実施
        #output = model(image)
        #_, prediction = torch.max(output, 1)
        #result = prediction[0].item()
        result = COCO_INSTANCE_CATEGORY_NAMES[predictions[0]['labels'].item()]

        return render_template("index.html", filepath=filepath, result=result)


if __name__ == "__main__":
    app.run(debug=True)