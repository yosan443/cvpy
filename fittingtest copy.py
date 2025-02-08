import cv2
import math
import numpy as np

def main():
    img = cv2.imread('test.png', cv2.IMREAD_COLOR)

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 黒no閾値
    threshold = 50

# 白塗り
    mask = hsv_image[:, :, 2] > threshold
    img[mask] = [255, 255, 255]

    # グレイスケール化
    gray1 = cv2.bitwise_and(img[:,:,0], img[:,:,1])
    gray1 = cv2.bitwise_and(gray1, img[:,:,2])

    # 二値化
    threshold_value = 10  # しきい値を指定
    _, binimg = cv2.threshold(gray1, threshold_value, 255, cv2.THRESH_BINARY)
    binimg = cv2.bitwise_not(binimg)

    # 黒の部分を灰色
    bimg = binimg // 4 + 255 * 3 //4
    resimg = cv2.merge((bimg,bimg,bimg)) 

    # 輪郭取得
    contours,hierarchy =  cv2.findContours(binimg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    max_size = 40
    min_size = 4 
    for i, cnt in enumerate(contours):
        # フィッティング
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            if min_size <= axes[0] <= max_size and min_size <= axes[1] <= max_size:
                print(ellipse)

                cx = int(ellipse[0][0])
                cy = int(ellipse[0][1])

                # 楕円描画
                resimg = cv2.ellipse(resimg, ellipse, (255, 0, 0), 2)
                cv2.drawMarker(resimg, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
                cv2.putText(resimg, str(i + 1), (cx + 3, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 1, cv2.LINE_AA)
            else:
                print("too large")
        else:
            print("too small")

    cv2.imshow('resimg',resimg)
    cv2.waitKey()

if __name__ == '__main__':
    main()