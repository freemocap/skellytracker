import cv2

from pathlib import Path
import onnxruntime as ort

from rtmlib import Wholebody, draw_skeleton

device = 'cuda'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
img = cv2.imread(r"D:\sfn\michael_wobble\backup_wobble_board_project_4-Aaron-2024-02-19 - Copy\labeled-data\Camera_001_synchronized\img002.png")

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)

keypoints, scores = wholebody(img)

# visualize

# if you want to use black background instead of original image,
# img_show = np.zeros(img_show.shape, dtype=np.uint8)

img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)


cv2.imshow('img', img_show)
cv2.waitKey()