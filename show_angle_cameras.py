import argparse
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

parse = argparse.ArgumentParser()
parse.add_argument('--vedio_path_a', type=str, default='assets/0528/2/2_h_1.mp4')
parse.add_argument('--vedio_path_b', type=str, default='assets/0528/2/2_l_1.mp4')

parse.add_argument('--angle', type=float, default=10)

parse.add_argument('--show_table', action='store_true')

parse.add_argument('--log_path_a', type=str, default='log/2_h_1.json')
parse.add_argument('--log_path_b', type=str, default='log/2_l_1.json')

def show(args):
    #read log and gt
    with open(args.log_path_a) as f:
        a_log_content = json.load(f)

    with open(args.log_path_b) as f:
        b_log_content = json.load(f)

    plt.xlim((0,30))
    plt.ylim(((-1) * np.pi / 2), np.pi / 2)
    plt.xlabel('Depth')
    plt.ylabel('Angle')

    last_a_frame_idx = 0
    last_b_frame_idx = 0
    a_depth_array = np.array([])
    a_angle_array = np.array([])
    a_raw_angle_array = np.array([])
    a_params = (0,0,0)

    b_depth_array = np.array([])
    b_angle_array = np.array([])
    b_raw_angle_array = np.array([])
    b_params = (0,0,0)

    num_less_than_25 = 0
    num_less_than_50 = 0
    num_less_than_75 = 0
    num_less_than_100 = 0
    num_more_than_100 = 0

    max_frame_idx = np.min([
        np.array([int(i) for i in a_log_content.keys()]).max(), 
        np.array([int(i) for i in b_log_content.keys()]).max()
    ])

    standard_delta_angle = args.angle

    for frame_idx in range(max_frame_idx):
        if str(frame_idx) in a_log_content:
            if len(np.array(a_log_content[str(frame_idx)]['depth'])) > 5:
                last_a_frame_idx = frame_idx
                a_depth_array = np.array(a_log_content[str(frame_idx)]['depth'])
                a_raw_angle_array = np.array(a_log_content[str(frame_idx)]['angle'])

                a_params = np.polyfit(a_depth_array, a_raw_angle_array, 2)

                a_angle_array = a_params[0] * a_depth_array * a_depth_array + a_params[1] * a_depth_array + a_params[2]

        a_depth_delta = (frame_idx - last_a_frame_idx) * 0.15
        a_depth_show_array = a_depth_array - a_depth_delta
        a_angle_show_array = a_raw_angle_array

        if str(frame_idx) in b_log_content:
            if len(np.array(b_log_content[str(frame_idx)]['depth'])) > 5:
                last_b_frame_idx = frame_idx
                b_depth_array = np.array(b_log_content[str(frame_idx)]['depth'])
                b_raw_angle_array = np.array(b_log_content[str(frame_idx)]['angle'])

                b_params = np.polyfit(b_depth_array, b_raw_angle_array, 2)

                b_angle_array = b_params[0] * b_depth_array * b_depth_array + b_params[1] * b_depth_array + b_params[2]

        b_depth_delta = (frame_idx - last_b_frame_idx) * 0.15
        b_depth_show_array = b_depth_array - b_depth_delta
        b_angle_show_array = b_raw_angle_array
        
        if args.show_table:
            plt.clf()
            plt.xlim((0,30))
            plt.ylim((-40, 40))
            plt.xlabel('Depth')
            plt.ylabel('Angle')

            # plt.plot(a_depth_show_array, a_angle_show_array, c='r')
            plt.plot(b_depth_show_array, b_angle_show_array, c='b')
            plt.pause(0.02)
           
        for i in range(0, 30):
            if not standard_delta_angle:
                break
            if len(a_depth_show_array) <= 0 or i <= a_depth_show_array.min() and i >= a_depth_show_array.max():
                continue
            if len(b_depth_show_array) <= 0 or i <= b_depth_show_array.min() and i >= b_depth_show_array.max():
                continue
                
            a_angle = a_params[0] * i * i + a_params[1] * i + a_params[2]
            b_angle = b_params[0] * i * i + b_params[1] * i + b_params[2]

            if np.sign(b_angle - a_angle) != np.sign(standard_delta_angle):
                num_more_than_100 += 1
                continue

            delta = (np.abs(b_angle - a_angle - standard_delta_angle)) / standard_delta_angle
            if delta < 0.25:
                num_less_than_25 += 1
            elif delta < 0.5:
                num_less_than_50 += 1
            elif delta < 0.75:
                num_less_than_75 += 1
            elif delta < 1:
                num_less_than_100 += 1
            else:
                num_more_than_100 += 1
    
    sum = num_less_than_25 + num_less_than_50 + num_less_than_75 + num_less_than_100 + num_more_than_100
    print("num: {}, {}, {}, {}, {}, {}".format(sum, num_less_than_25, num_less_than_50, num_less_than_75, num_less_than_100, num_more_than_100))
    print("rate: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(num_less_than_25 / sum, num_less_than_50 / sum, num_less_than_75 / sum, num_less_than_100 / sum, num_more_than_100 / sum))
    

if __name__ == '__main__':
    show(parse.parse_args())

