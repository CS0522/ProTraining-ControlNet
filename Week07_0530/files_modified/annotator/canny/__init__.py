import cv2


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        print("[annotator/canny/__init__.py] 调用Canny边缘检测预处理器函数")
        return cv2.Canny(img, low_threshold, high_threshold)
