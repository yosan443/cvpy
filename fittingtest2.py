import cv2
import math
import numpy as np

def main():
    img = cv2.imread('fitdelta/test2.png', cv2.IMREAD_COLOR)

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 黒no閾値
    threshold = 50

    # 白塗り
    mask = hsv_image[:, :, 2] > threshold
    img[mask] = [255, 255, 255]

    # グレースケール化
    gray1 = cv2.bitwise_and(img[:,:,0], img[:,:,1])
    gray1 = cv2.bitwise_and(gray1, img[:,:,2])

    # 二値化
    threshold_value = 10  # しきい値を指定
    _, binimg = cv2.threshold(gray1, threshold_value, 255, cv2.THRESH_BINARY)
    binimg = cv2.bitwise_not(binimg)

    # 楕円検出とラベリング
    ellipses1 = detect_and_label_ellipses(cv2.imread('fitdelta/test1.png'))
    ellipses2 = detect_and_label_ellipses(img)

    # マッチしない楕円をリストする
    unmatched_ellipses = match_ellipses(ellipses1, ellipses2)

    # 楕円の最大サイズを設定
    max_size = 100

    # マッチしない楕円を表示
    for ellipse in unmatched_ellipses:
        (x, y), (MA, ma), angle = ellipse
        if 0 < MA < max_size and 0 < ma < max_size:  # 楕円のサイズが有効かどうかをチェック
            cv2.ellipse(img, ellipse, (0, 0, 255), 2)

    # 結果の画像を表示
    cv2.imshow('Unmatched Ellipses', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

if __name__ == "__main__":
    main()