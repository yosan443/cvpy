import cv2
import numpy as np

def detect_and_label_ellipses(image):
    # グレースケール画像に変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Cannyエッジ検出を適用
    edges = cv2.Canny(gray, 50, 150)
    
    # 輪郭を検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 検出された楕円のバウンディングボックスを取得
    ellipses = []
    for contour in contours:
        if len(contour) >= 5: # 楕円フィッティングには少なくとも5つの点が必要
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
    
    return ellipses

def match_ellipses(ellipses1, ellipses2):
    # マッチしない楕円をリストする
    unmatched_ellipses = [ellipse for ellipse in ellipses1 if ellipse not in ellipses2]
    return unmatched_ellipses

# 1つ目の画像を読み込む
image1 = cv2.imread('masktest.png')
# 2つ目の画像を読み込む
image2 = cv2.imread('test.png')

# 楕円を検出してラベリング
ellipses1 = detect_and_label_ellipses(image1)
ellipses2 = detect_and_label_ellipses(image2)

# マッチしない楕円をリストする
unmatched_ellipses = match_ellipses(ellipses1, ellipses2)

# マッチしない楕円を表示
for ellipse in unmatched_ellipses:
    (x, y), (MA, ma), angle = ellipse
    cv2.ellipse(image1, ellipse, (0, 0, 255), 2)

# 結果の画像を表示
cv2.imshow('Unmatched Ellipses', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
