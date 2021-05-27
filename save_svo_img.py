########################################################################
#
# Copyright (c) 2021, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Read SVO sample to read the video and the information of the camera. It can pick a frame of the svo and save it as
    a JPEG or PNG file. Depth map and Point Cloud can also be saved into files.
"""
import json
import sys
import numpy as np
import pyzed.sl as sl
import os
import cv2
from tqdm import tqdm

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def main():
    if len(sys.argv) != 2:
        print("Please specify path to .svo file.")
        exit()

    file_path = sys.argv[1]
    print("Reading SVO file: {0}".format(file_path))

    file_name = os.path.basename(file_path)
    image_folder = os.path.join("assets/svo_data", file_name.split('.')[0])
    image_index = 0
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)


    input_type = sl.InputType()
    input_type.set_from_svo_file(file_path)
    init = sl.InitParameters(input_t=input_type)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    left_mat = sl.Mat()
    right_mat = sl.Mat()

    save_camera_information(cam, image_folder)
    total = cam.get_svo_number_of_frames()
    print()
    for i in tqdm(range(total)):
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(left_mat, view=sl.VIEW.LEFT)
            cam.retrieve_image(right_mat, view=sl.VIEW.RIGHT)
            left_file_path = os.path.join(image_folder, "{}_l.jpg".format(i))
            right_file_path = os.path.join(image_folder, "{}_r.jpg".format(i))
            # cv2.imshow("zed", left_mat.get_data())
            cv2.imwrite(left_file_path, left_mat.get_data())
            cv2.imwrite(right_file_path, right_mat.get_data())
        else:
            break

    cam.close()
    print("\nFINISH")


def save_camera_information(cam, image_folder):

            
    calibration_parameters = cam.get_camera_information().calibration_parameters.left_cam
    fx = calibration_parameters.fx
    fy = calibration_parameters.fy
    cx = calibration_parameters.cx
    cy = calibration_parameters.cy
    camera_mtx = [
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ]
    camera_dist = [calibration_parameters.disto.tolist()]
    camera_param = {}
    camera_param["left_camera"] = {
        "camera_mtx": camera_mtx,
        "camera_dist": camera_dist
    }

    calibration_parameters = cam.get_camera_information().calibration_parameters.right_cam
    fx = calibration_parameters.fx
    fy = calibration_parameters.fy
    cx = calibration_parameters.cx
    cy = calibration_parameters.cy
    camera_mtx = [
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ]
    camera_dist = [calibration_parameters.disto.tolist()]
    camera_param["right_camera"] = {
        "camera_mtx": camera_mtx,
        "camera_dist": camera_dist
    }

    camera_param["FPS"] = cam.get_camera_information().camera_fps
    camera_param["resolution"] = [round(cam.get_camera_information().camera_resolution.width, 2), cam.get_camera_information().camera_resolution.height]
    
    stereo = np.zeros((4,4))
    stereo[0:3,0:3] = eulerAnglesToRotationMatrix(cam.get_camera_information().calibration_parameters.R)
    stereo[0:3,2:3] = np.matrix(cam.get_camera_information().calibration_parameters.T).T
    
    camera_param["stereo"] = [stereo.tolist()]
    # print(cam.get_camera_information().calibration_parameters.R)
    # print(cam.get_camera_information().calibration_parameters.T)
    camera_param["frame_num"] = cam.get_svo_number_of_frames()

    config_path = os.path.join(image_folder, "config.json")

    with open(config_path, 'w') as f:
        json.dump(camera_param, f, indent=4)

    print("Frame count: {0}.\n".format(cam.get_svo_number_of_frames()))

if __name__ == "__main__":
    main()