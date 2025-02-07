import cv2
import numpy as np

# 画像を読み込む
image = cv2.imread('test.png')

# 画像をHSV色空間に変換
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 黒さの閾値を設定 (V値が50以下を黒とみなす)
threshold = 50

# V値が閾値以下のピクセルを白に変換
mask = hsv_image[:, :, 2] > threshold
image[mask] = [255, 255, 255]

# 変換後の画像を保存または表示
cv2.imwrite('output_image.jpg', image)
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()