'''
box[0]: [[163  32]]右上
box[1]: [[63 72]]   左上
box[2]: [[150 215]]左下
box[3]: [[268 144]]右下
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

start = time.time()
# img = cv2.imread("/home/dzc/Data/zed/raw/1/1.jpg")
img = Image.open("/home/dzc/Data/zed/raw/1/1.jpg")
img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
ROTATED_SIZE_X  = 800 #透视变换后的表盘图像大小
ROTATED_SIZE_Y  = 449 #透视变换后的表盘图像大小
CUT_SIZE     =  0   #透视变换时四周裁剪长度

W_cols, H_rows= img.shape[:2]
# print(H_rows, W_cols)


# 原图中书本的四个角点(左上、右上、右下、左下),与变换后矩阵位置,排好序的角点输出，0号是左上角，顺时针输出
pts1 = np.float32([[-200, 400], [700, 50], [1100, 90], [500, 1100]])
#变换后矩阵位置
pts2 = np.float32([[0, 0],[ROTATED_SIZE_X,0],[ROTATED_SIZE_X, ROTATED_SIZE_Y],[0,ROTATED_SIZE_Y],]) # 可以换成别的点


# 生成透视变换矩阵；进行透视变换
M = cv2.getPerspectiveTransform(pts1, pts2)
print(M)
dst = cv2.warpPerspective(img, M, (ROTATED_SIZE_X,ROTATED_SIZE_Y))
end = time.time()

print(end - start)

cv2.imshow("original_img",img)
cv2.imshow("result",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()