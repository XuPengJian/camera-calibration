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

# 标定板【内角点】的个数
CHECKERBOARD = (9, 6)
# 每个格子的实际单位大小,cm为单位
SCALAR = 6.95
# 角点优化、迭代的终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 图片所在路径
img_file = r'D:\gitlab\camera_calibration\rebuild\192.168.8.254_01_20230412124713773.jpg'

# step1:定义标定板在真实世界的坐标
# 创建一个list来保存每张图片中角点的3D坐标
objpoints = []
# 创建一个list来保存每张图片中角点的2D坐标
imgpoints = []

# 定义3D坐标：[row, col, z]
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

img = cv2.imread(img_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# step3:计算标定板的角点的2D坐标
# 寻找角点坐标，如果找到ret返回True，corners:[col, row]，原点在左上角
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                         cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

if ret == True:
    objpoints.append(objp)
    # 调用cornerSubpix对2D角点坐标位置进行优化
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)
    # 绘制寻找到的角点，从红色开始绘制，紫色结束
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    retval, rvec, tvec = cv2.solvePnP(objpoints[0], imgpoints[0], mtx, dist)
else:
    print(f"图片识别失败")

print('objpoints:', objpoints[0])
print('imgpoints:', imgpoints[0])


print("旋转矩阵:")
print(rvec, "\n")
print("平移矩阵:")
print(tvec, "\n")

# 测试验证，在cv2中画出来

# 定义世界坐标系下的三维点
# x轴红色
x_point_3d = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [40.0, 0.0, 0.0]])/SCALAR
# y轴绿色
y_point_3d = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 20.0, 0.0], [0.0, 30.0, 0.0], [0.0, 40.0, 0.0]])/SCALAR
# z轴蓝色
z_point_3d = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 20.0], [0.0, 0.0, 30.0], [0.0, 0.0, 40.0]])/SCALAR

# 使用projectPoints函数将三维点投影到二维图像平面
# x轴
x_point_2d, _ = cv2.projectPoints(x_point_3d, rvec, tvec, mtx, dist)
x_point_2d = np.round(x_point_2d).astype(np.int32)
print(x_point_2d)

# y轴
y_point_2d, _ = cv2.projectPoints(y_point_3d, rvec, tvec, mtx, dist)
y_point_2d = np.round(y_point_2d).astype(np.int32)
print(y_point_2d)

# z轴
z_point_2d, _ = cv2.projectPoints(z_point_3d, rvec, tvec, mtx, dist)
z_point_2d = np.round(z_point_2d).astype(np.int32)
print(z_point_2d)

# 在cv2中画出这个点
# 定义要绘制的圆的参数
r = 5

# x轴红色
for point in x_point_2d:
    color = (0, 0, 255)  # 红色
    x, y = point[0]
    # 在图像上画一个圆
    cv2.circle(img, (x, y), r, color, -1)
point_2d = x_point_2d.reshape((1, -1, 2))
cv2.polylines(img, point_2d, False, (0, 0, 255), thickness=2)

# y轴绿色
for point in y_point_2d:
    color = (0, 255, 0)  # 绿色
    x, y = point[0]
    # 在图像上画一个圆
    cv2.circle(img, (x, y), r, color, -1)
point_2d = y_point_2d.reshape((1, -1, 2))
cv2.polylines(img, point_2d, False, (0, 255, 0), thickness=2)

# z轴蓝色
for point in z_point_2d:
    color = (255, 0, 0)  # 蓝色
    x, y = point[0]
    # 在图像上画一个圆
    cv2.circle(img, (x, y), r, color, -1)
point_2d = z_point_2d.reshape((1, -1, 2))
cv2.polylines(img, point_2d, False, (255, 0, 0), thickness=2)

# 显示图像
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
