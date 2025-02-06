import cv2
import numpy as np
import os

# 画像を読み込む
image = cv2.imread('test.png')

# グレースケールに変換する
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# エッジ検出を行う
edges = cv2.Canny(gray, 50, 150)

# 輪郭を検出する
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# outputsフォルダが存在しない場合は作成する
os.makedirs('outputs', exist_ok=True)

# 各輪郭を矩形として認識し、画像を分割して保存する
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    
    # 矩形のサイズが10×90ピクセル以下の場合はスキップ
    if w <= 90 or h <= 10:
        continue

    if w > 110 or h > 20:
        continue
    
    box_image = image[y:y+h, x:x+w]
    
    # 矩形領域の内部に空白があるかをチェックする
    box_gray = gray[y:y+h, x:x+w]
    _, box_thresh = cv2.threshold(box_gray, 240, 255, cv2.THRESH_BINARY)
    white_pixels = cv2.countNonZero(box_thresh)
    total_pixels = box_thresh.size
    
    # 空白が一定割合以上ある場合のみ保存する
    if white_pixels / total_pixels <= 0.1:  # ここでは10%を閾値としています
        continue
    
    # 矩形領域の内部に円が含まれているかをチェックする
    circles = cv2.HoughCircles(box_gray, cv2.HOUGH_GRADIENT, dp=0.5, minDist=1, param1=50, param2=20, minRadius=1, maxRadius=50)
    
    if circles is None:
        continue
    
    # 円が検出された場合のみ保存する
    cv2.imwrite(f'outputs/box_{i}.jpg', box_image)

print("画像の分割が完了しました。")