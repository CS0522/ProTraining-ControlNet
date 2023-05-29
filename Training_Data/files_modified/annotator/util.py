print("[annotator/util.py] 开始导入依赖库")
import numpy as np
import cv2
import os

print("[annotator/util.py] 开始加载预处理器(边缘检测)模型")
annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')


def HWC3(x):
    print("[annotator/util.py] HWC3函数处理图片")
    assert x.dtype == np.uint8
    if x.ndim == 2:
        print("[annotator/util.py] HWC3函数位置一")
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    print("[annotator/util.py] HWC3函数位置二")
    if C == 3:
        print("[annotator/util.py] HWC3函数位置三")
        return x
    if C == 1:
        print("[annotator/util.py] HWC3函数位置四")
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        print("[annotator/util.py] HWC3函数位置五")
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        print("[annotator/util.py] HWC3函数位置六")
        return y


def resize_image(input_image, resolution):
    print("[annotator/util.py] 开始缩放图像大小")
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    print("[annotator/util.py] 图像大小缩放完成")
    return img
