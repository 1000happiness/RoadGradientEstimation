import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parse = argparse.ArgumentParser()
parse.add_argument('--vedio_path', type=str, default='assets/zed1.mkv')
parse.add_argument('--min_frame_idx', type=int, default=-1)
parse.add_argument('--max_frame_idx', type=int, default=-1)

parse.add_argument('--pitch_delta', type=float, default=0)

parse.add_argument('--show_image', action='store_true')
parse.add_argument('--show_table', action='store_true')

parse.add_argument('--log_path', type=str, default='log/zed3.json')
parse.add_argument('--gt_path', type=str, default='log/zed3_gt.csv')

def show(args):
    #read log and gt
    with open(args.log_path) as f:
        log_content = json.load(f)

    plt.xlim((0,30))
    plt.ylim(((-1) * np.pi / 2), np.pi / 2)
    plt.xlabel('Depth')
    plt.ylabel('Angle')

    gt_content = pd.read_csv(args.gt_path)
    change_lo_la_to_distance(gt_content)

    front_row_index = 0
    front_distance = 0

    gt_depth_list = []
    gt_angle_list = []
    last_log_depth_array = np.array([])
    last_log_angle_array = np.array([])
    raw_log_depth_array = np.array([])
    raw_log_angle_array = np.array([])
    params = (0,0,0)
    last_log_frame_idx = 0

    num_less_than_2_5 = 0
    num_less_than_5 = 0
    num_less_than_7_5 = 0
    num_less_than_10 = 0
    num_more_than_10 = 0

    total_delta = 0

    for row_idx, row in gt_content.iterrows():
        if args.min_frame_idx > 0 and row['frame_idx'] < args.min_frame_idx:
            continue

        if args.max_frame_idx > 0 and row['frame_idx'] > args.max_frame_idx:
            break

        ret, front_row_index, front_distance = find_x_meters_front_row_idx(30, gt_content, row_idx, 0)
        front_distance -= row['distance']
        if not ret:
            break

        gt_depth_list = []
        for i in range(row_idx, front_row_index + 1):
            depth = gt_content.loc[row_idx: i]['distance'].sum()
            gt_depth_list.append(depth)

        gt_angle_list = gt_content.loc[row_idx: front_row_index]['pitch'].tolist()

        if str(int(row['frame_idx'])) in log_content:
            last_log_frame_idx = row['frame_idx']
            raw_log_depth_array = np.array(log_content[str(int(row['frame_idx']))]['depth'])
            raw_log_angle_array = np.array(log_content[str(int(row['frame_idx']))]['angle'])

            params = np.polyfit(raw_log_depth_array, raw_log_angle_array, 2)

            last_log_depth_array = raw_log_depth_array
            last_log_angle_array = params[0] * raw_log_depth_array * raw_log_depth_array + params[1] * raw_log_depth_array + params[2] + args.pitch_delta

        depth_delta = gt_content[(gt_content['frame_idx'] > last_log_frame_idx) & (gt_content['frame_idx'] <= row['frame_idx'])]['distance'].sum()

        log_depth_array = last_log_depth_array - depth_delta
        log_angle_list = last_log_angle_array

        for i in range(0, 30):
            if len(log_depth_array) > 0 and i > log_depth_array.min() and i < log_depth_array.max():
                gt_angle = get_pitch_by_depth(gt_content, row_idx, gt_depth_list, i)

                # pred_angle = params[0] * i * i + params[1] * i + params[2] + args.pitch_delta
                # delta = np.abs(pred_angle - gt_angle)

                raw_pred_angle = get_angle_by_depth(log_depth_array, raw_log_angle_array, i) + args.pitch_delta
                delta = np.abs(raw_pred_angle - gt_angle)

                total_delta += delta
                if delta < 2.5:
                    num_less_than_2_5 += 1
                elif delta < 5:
                    num_less_than_5 += 1
                elif delta < 7.5:
                    num_less_than_7_5 += 1
                elif delta < 10:
                    num_less_than_10 += 1
                else:
                    num_more_than_10 += 1

        if args.show_table:
            plt.clf()
            plt.xlim((0,30))
            plt.ylim((-15, 15))
            plt.xlabel('Depth')
            plt.ylabel('Angle')

            plt.plot(gt_depth_list, gt_angle_list, c='r')
            plt.plot(log_depth_array, log_angle_list, c='b')
            plt.plot(log_depth_array, raw_log_angle_array + args.pitch_delta, c='g')
            plt.pause(0.033)

        

        print(row['frame_idx'], num_less_than_2_5, num_less_than_5, num_less_than_7_5, num_less_than_10, num_more_than_10, sep="| ")
    
    sum = num_less_than_2_5 + num_less_than_5 + num_less_than_7_5 + num_less_than_10 + num_more_than_10
    print("average delta:", total_delta / sum)
    print("num: {}, {}, {}, {}, {}, {}".format(sum, num_less_than_2_5, num_less_than_5, num_less_than_7_5, num_less_than_10, num_more_than_10))
    print("rate: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(num_less_than_2_5 / sum, num_less_than_5 / sum, num_less_than_7_5 / sum, num_less_than_10 / sum, num_more_than_10 / sum))

    # for frame_idx in log_content:
    #     plt.clf()
    #     plt.xlim((0,30))
    #     plt.ylim(((-1) * np.pi / 2), np.pi / 2)
    #     plt.xlabel('Depth')

    #     plt.ylabel('Angle')

    #     depth_list = log_content[frame_idx]['depth']
    #     angle_list = log_content[frame_idx]['angle']
        
    #     plt.plot(depth_list, angle_list, c='r',ls='-', marker='o', mec='b',mfc='w')
    #     plt.pause(0.1)

def change_lo_la_to_distance(gt_content):
    gt_content['distance'] = 0
    for index, row in gt_content.iterrows():
        if index < 1:
            continue
        lo_delta = (gt_content.at[index, 'longitude'] - gt_content.at[index - 1, 'longitude']) * 85390
        la_delta = (gt_content.at[index, 'latitude'] - gt_content.at[index - 1, 'latitude']) * 85390
        gt_content.loc[index, 'distance'] = np.sqrt(lo_delta * lo_delta + la_delta * la_delta)  #meter

def find_x_meters_front_row_idx(x, gt_content, base_row_idx, base_distance):
    if base_distance > x:
        return True, base_row_idx, base_distance

    current_row_idx = base_row_idx + 1
    current_distance = base_distance
    while current_row_idx < len(gt_content):
        current_distance += gt_content.loc[current_row_idx]['distance']
        if current_distance > x:
            return True, current_row_idx, current_distance
        current_row_idx += 1
    
    return False, None, None

def get_pitch_by_depth(gt_content, base_idx, depth_list, depth):
    depth_list = np.array(depth_list)
    delta_idx = np.where(depth_list < depth)[0][0]
    return gt_content.loc[base_idx + delta_idx]['pitch']

def get_angle_by_depth(log_depth_array, raw_log_angle_array, i):
    if (log_depth_array == i).any():
        return raw_log_angle_array[log_depth_array == i]
    
    left_depth = log_depth_array[log_depth_array < i][-1]
    right_depth = log_depth_array[log_depth_array > i][0]
    left_angle = raw_log_angle_array[log_depth_array < i][-1]
    right_angle = raw_log_angle_array[log_depth_array > i][0]
    return (right_angle - left_angle) / (right_depth - left_depth) * (i - left_depth) + left_angle
    

if __name__ == '__main__':
    show(parse.parse_args())

