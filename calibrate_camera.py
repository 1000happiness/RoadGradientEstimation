import cv2  
import numpy as np  
import glob  
import json
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--cali_img_folder', type=str, default='assets/calibration_data/zed/')
parse.add_argument('--cali_img_suffix', type=str, default='.jpg')
parse.add_argument('--camera_config_path', type=str, default='config/camera.json')
  
def calibrateIntrinsicParam(args):
    shape = (10, 4)

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001  
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)  
    
    # 获取标定板角点的位置  
    objp = np.zeros((shape[1]*shape[0],3), np.float32)  
    objp[:,:2] = np.mgrid[0:shape[0],0:shape[1]].T.reshape(-1,2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y  

    obj_points = []    # 存储3D点  
    img_points = []    # 存储2D点  
    
    images = glob.glob('{}*{}'.format(args.cali_img_folder, args.cali_img_suffix))
    size = None  
    for fname in images:  
        img = cv2.imread(fname)  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        size = gray.shape[::-1]  
        ret, corners = cv2.findChessboardCorners(gray, (shape[0],shape[1]), None)  
        if ret:  
            obj_points.append(objp)  
            corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)  # 在原角点的基础上寻找亚像素角点  
            if corners2.any():  
                img_points.append(corners2)  
            else:  
                img_points.append(corners)  

    if len(img_points) != 0:
        # 标定  
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)  
        print ('ret:',ret)  
        print ('mtx:\n',mtx)        # 内参数矩阵  
        print ('dist:\n',dist)      # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)  
        # print ('rvecs:\n',rvecs)    # 旋转向量  # 外参数  
        # print ('tvecs:\n',tvecs)    # 平移向量  # 外参数  
        camera_param = {
            'camera_mtx': mtx.tolist(),
            'camera_dist': dist.tolist()
        }

        print(camera_param)

        with open(args.camera_config_path,'w') as f:
            json.dump(camera_param, f, indent=4)

def main():
    print("开始进行内参标定，标定图片为预先准备的棋盘图片")
    calibrateIntrinsicParam(parse.parse_args())
    print("内参标定完成")

if __name__ == '__main__':
    main()


