import os.path
import cv2
import numpy as np
from numpy import array
import glob

"""
超参获取
"""
# 相机内参
mtx = np.array([[1.72587767e+03, 0.00000000e+00, 9.38261878e+02],
                [0.00000000e+00, 1.74511341e+03, 5.26798554e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# 畸变参数
dist = np.array([[-0.92145838, 1.18408159, 0.00207916, 0.00441199, -0.76582526]])

# 去畸变
images = glob.glob('rebuild/*.jpg')
images = sorted(images)
for i, img_file in enumerate(images):
    img = cv2.imread(img_file)
    img_name = os.path.basename(img_file)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 使用undistort矫正图像
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # ROI超参
    x, y, w, h = roi
    off_x = 0
    off_y = 0

    # 图片裁剪
    dst2 = dst[y - off_y:y + h + off_y, x - off_x:x + w + off_x]
    # cv2.imshow("original image", img)
    # cv2.imshow("image undistored", dst)
    # cv2.imshow("image undistorted ROI", dst2)
    cv2.imwrite('undistorted/' + img_name, dst2)