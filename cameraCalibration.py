import os.path
import cv2
import numpy as np
import glob

# 标定板【内角点】的个数
CHECKERBOARD = (9, 6)

# 角点优化、迭代的终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# step1:定义标定板在真实世界的坐标
# 创建一个list来保存每张图片中角点的3D坐标
objpoints = []
# 创建一个list来保存每张图片中角点的2D坐标
imgpoints = []

# 定义3D坐标：[row, col, z]
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# step2:提取不同角度拍摄的图片
images = glob.glob('images/*.jpg')
images = sorted(images)
for i, fname in enumerate(images):
    img = cv2.imread(fname)
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
        cv2.imshow(fname + " succeed", img)
    else:
        print(f"第{i}张图，{fname}未发现足够角点")
        cv2.imshow(fname + " failed", img)
    cv2.waitKey(1)
cv2.destroyAllWindows()

print('objpoints:')
print(objpoints)
print('imgpoints:')
print(imgpoints)
h, w = img.shape[:2]

# step4: 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("相机内参:")  # [[fx, 0, cx],[0, fy, cy],[0, 0, 1]]
print(mtx, "\n")
print("畸变参数:")  # k1,k2,p1,p2,k3
print(dist, "\n")
# print("旋转矩阵:")
# print(rvecs, "\n")
# print("平移矩阵:")
# print(tvecs, "\n")

# step5: 去畸变
img_file = r'D:\gitlab\camera_calibration\images\192.168.8.254_01_2023041110021450.jpg'
img_name = os.path.basename(img_file) + '.jpg'
img = cv2.imread(img_file)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 使用undistort矫正图像
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# 图片裁剪
x, y, w, h = roi
dst2 = dst[y:y + h, x:x + w]
print(f"ROI: x:{x}, y:{y}, w:{w}, h:{h}")
# cv2.imshow("original image", img)
# cv2.imshow("image undistored", dst)
# cv2.imshow("image undistorted ROI", dst2)
cv2.imwrite('undistorted/' + img_name, dst)

# # 使用remapping校正图像
# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
# dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# # 图片裁剪
# x, y, w, h = roi
# dst2 = dst[y:y+h, x:x+w]
# cv2.imshow("original image", img)
# cv2.imshow("image undistored", dst)
# cv2.imshow("image undistorted ROI", dst2)
# cv2.imwrite('undistorted/'+img_name, dst2)

# step6: 计算重投影误差
mean_error = 0
for i in range(len(objpoints)):
    # 使用内外参和畸变参数对点进行重投影
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)  # L2范数，均方误差
    mean_error += error

mean_error /= len(objpoints)
print("total error: {}".format(mean_error))
