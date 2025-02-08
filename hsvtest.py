import cv2
import numpy as np

class MarkSheetDetector:
    def __init__(self, threshold=127):
        self.threshold = threshold

    def preprocess_image(self, image):
        """画像の前処理を行う"""
        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 二値化
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        return binary

    def find_grid(self, binary):
        """マークシートのグリッドを検出する"""
        # 輪郭を検出
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 面積でフィルタリング
        min_area = 100  # 最小面積（調整可能）
        grid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        return grid_contours

    def detect_marked_cells(self, image):
        """塗られたマスを検出する"""
        # 前処理
        binary = self.preprocess_image(image)
        
        # グリッドを検出
        grid_contours = self.find_grid(binary)
        
        # 結果を格納するリスト
        marked_cells = []
        
        # 各セルを解析
        for i, contour in enumerate(grid_contours):
            # 輪郭の矩形を取得
            x, y, w, h = cv2.boundingRect(contour)
            
            # セル内の塗りつぶし率を計算
            cell_roi = binary[y:y+h, x:x+w]
            fill_ratio = np.sum(cell_roi == 255) / (w * h)
            
            # 塗りつぶし率が閾値を超えていれば、マークされているとみなす
            if fill_ratio > 0.5:  # 閾値は調整可能
                marked_cells.append({
                    'cell_id': i,
                    'position': (x, y),
                    'size': (w, h),
                    'fill_ratio': fill_ratio
                })
        
        return marked_cells

    def visualize_results(self, image, marked_cells):
        """結果を可視化する"""
        result = image.copy()
        
        # 検出されたセルを矩形で囲む
        for cell in marked_cells:
            x, y = cell['position']
            w, h = cell['size']
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 塗りつぶし率を表示
            text = f"{cell['fill_ratio']:.2f}"
            cv2.putText(result, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result

def process_mark_sheet(image_path):
    """マークシートを処理する主要な関数"""
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("画像を読み込めませんでした")
    
    # マークシート検出器を初期化
    detector = MarkSheetDetector()
    
    # マークされたセルを検出
    marked_cells = detector.detect_marked_cells(image)
    
    # 結果を可視化
    result_image = detector.visualize_results(image, marked_cells)
    
    return marked_cells, result_image

# 使用例
if __name__ == "__main__":
    image_path = "test.png"  # 画像のパスを指定
    try:
        marked_cells, result_image = process_mark_sheet(image_path)
        
        # 結果を表示
        print(f"検出されたマークの数: {len(marked_cells)}")
        for cell in marked_cells:
            print(f"セルID: {cell['cell_id']}, 位置: {cell['position']}, "
                  f"サイズ: {cell['size']}, 塗りつぶし率: {cell['fill_ratio']:.2f}")
        
        # 結果画像を保存
        cv2.imwrite("result.jpg", result_image)
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")