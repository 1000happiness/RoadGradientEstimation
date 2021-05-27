import argparse
import json
import time
import cv2
import os
from PIL import Image
import numpy as np

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import torch.nn.functional as F

from nets.bisenet_v2 import BiSeNetV2
from nets.monodepth_v2 import ResnetEncoder, DepthDecoder
from utils.monodepth_utils import disp_to_depth
from utils.bisenet_utils import ToTensor
from utils.reprojection import reprojection_to_3D, BackprojectDepth
from utils.plane_utils import get_plane_by_lstsq, get_plane_by_ransac, get_plane2Zaxis_angle
from utils.plot_utils import show_3d_points
from utils.img_utils import color_lane_line

parse = argparse.ArgumentParser()
parse.add_argument('--img_path', type=str, default='assets/1.jpeg')

parse.add_argument('--no_cuda', type=int, default=0)

parse.add_argument('--camera_config_path', type=str, default='config/camera.json')
parse.add_argument('--monodepth_config_path', type=str, default='config/monodepth_v2.json')
parse.add_argument('--bisenet_config_path', type=str, default='config/bisenet_v2.json')

parse.add_argument('--undistort_flag', type=int, default=1)
parse.add_argument('--color_lane_line_flag', type=int, default=1)
parse.add_argument('--invalid_mask_path', type=str, default="")

parse.add_argument('--show_image', action="store_true")

def main(args):
    print('Predict on {}'.format(args.img_path))
    raw_img_cv = cv2.imread(args.img_path)
    original_height, original_width, _ = raw_img_cv.shape

    # cuda or cpu
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load camera config
    with open(args.camera_config_path) as f:
        camera_config = json.load(f)
    camera_mtx = np.matrix(camera_config['camera_mtx'])
    camera_dist = np.matrix(camera_config['camera_dist'])
    
     # load invalid mask
    if args.invalid_mask_path:
        invalid_mask_cv = cv2.imread(args.invalid_mask_path)
        invalid_mask_np = np.array(cv2.cvtColor(invalid_mask_cv, cv2.COLOR_RGB2GRAY))
        valid_mask_np = (invalid_mask_np == 0).astype(np.uint8)
        # valid_mask_torch = torch.from_numpy(valid_mask_np).to(device)

    # preprocess img
    if args.undistort_flag:
        new_camera_mtx, camera_roi = cv2.getOptimalNewCameraMatrix(camera_mtx, camera_dist, (original_width, original_height), 1, (original_width, original_height))
        inv_camera_mtx = np.eye(4, dtype=np.float32)
        inv_camera_mtx[:3,:3] = np.linalg.pinv(new_camera_mtx)
        inv_camera_mtx_torch = torch.from_numpy(inv_camera_mtx).to(device)
        roi_x, roi_y, roi_w, roi_h = camera_roi
        original_height, original_width = roi_h, roi_w
        raw_img_cv = cv2.undistort(raw_img_cv, camera_mtx, camera_dist, None, new_camera_mtx)
        raw_img_cv = raw_img_cv[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
        if args.invalid_mask_path:
            valid_mask_np = valid_mask_np[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
    else:
        inv_camera_mtx = np.eye(4, dtype=np.float32)
        inv_camera_mtx[:3,:3] = np.linalg.pinv(camera_mtx)
        inv_camera_mtx_torch = torch.from_numpy(inv_camera_mtx).to(device)

    # back_project_depth
    back_project = BackprojectDepth(1, original_height, original_width)
    back_project.to(device)

    # init bisenet_v2 network
    print('Start initializing bisenet_v2 network')
    with open(args.bisenet_config_path) as f:
        bisenet_config = json.load(f)
    bisenet = BiSeNetV2(bisenet_config['n_classes'])
    bisenet.load_state_dict(torch.load(bisenet_config['model_path'], map_location='cpu'))
    bisenet.to(device)
    bisenet.eval()

    ground_class_number = bisenet_config['ground_class_number']
    bisenet_to_tensor = ToTensor(
        device=device,
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )
    print('Finish initializing bisenet_v2 network')

    # init monodepth_v2 network
    print('Start initializing monodepth network')
    with open(args.monodepth_config_path) as f:
        monodepth_config = json.load(f)

    encoder_path = os.path.join(monodepth_config['model_path'], 'encoder.pth')
    monodepth_encoder = ResnetEncoder(monodepth_config['resnet_layer_number'], False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    feed_height, feed_width = loaded_dict_enc['height'], loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in monodepth_encoder.state_dict()}
    monodepth_encoder.load_state_dict(filtered_dict_enc)
    monodepth_encoder.to(device)
    monodepth_encoder.eval()
    
    depth_decoder_path = os.path.join(monodepth_config['model_path'], 'depth.pth')
    monodepth_decoder = DepthDecoder(num_ch_enc=monodepth_encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    monodepth_decoder.load_state_dict(loaded_dict)
    monodepth_decoder.to(device)
    monodepth_decoder.eval()
    print('Finish initializing monodepth network')

    with torch.no_grad():
        # raw_img to bisenet input
        resized_img_cv = cv2.resize(raw_img_cv, (1024, 512))
        bisenet_input_torch = bisenet_to_tensor(resized_img_cv).unsqueeze(0).to(device)
        
        # bisenet process
        bisenet_output_torch = bisenet(bisenet_input_torch)[0].argmax(dim=1).squeeze()

        # bisenet_output(torch) to ground_mask(np)
        ground_mask_np = (bisenet_output_torch.cpu().numpy() == ground_class_number).astype(np.uint8)
        ground_mask_np = cv2.resize(ground_mask_np, (original_width, original_height))
        ground_mask_torch = torch.from_numpy(ground_mask_np).to(device)
        if args.invalid_mask_path:
            ground_mask_np = ground_mask_np * valid_mask_np

        # ground_mask(np) to bisenet_colored_output(cv)
        if args.show_image:
            bisenet_colored_cv = cv2.cvtColor(ground_mask_np * 255, cv2.COLOR_GRAY2RGB)
        if args.color_lane_line_flag:
            ground_img_cv = raw_img_cv * cv2.cvtColor(ground_mask_np, cv2.COLOR_GRAY2RGB)
            ground_img_cv = color_lane_line(ground_img_cv)
            raw_img_cv = raw_img_cv * cv2.cvtColor((ground_mask_np != 1).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            raw_img_cv = raw_img_cv + ground_img_cv

        # raw_img(cv) to monodepth_input(torch)
        raw_img_pil = Image.fromarray(raw_img_cv)
        resized_image_pil = raw_img_pil.resize((feed_width, feed_height), Image.LANCZOS)
        monodepth_input_torch = transforms.ToTensor()(resized_image_pil).unsqueeze(0).to(device)

        # monodepth process
        features = monodepth_encoder(monodepth_input_torch)
        disp_torch = monodepth_decoder(features)[("disp", 0)]

        # monodepth_output(torch) to colored_img(cv)
        disp_torch = F.interpolate(disp_torch, (original_height, original_width), mode="bilinear", align_corners=False)
        if args.show_image:
            disp_resized_np = disp_torch.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            monodepth_colored_output = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            monodepth_colored_output_cv = cv2.cvtColor(monodepth_colored_output, cv2.COLOR_RGB2BGR)
        
        # monodepth_output(torch) to depth_output(torch)
        _ , depth_output_torch = disp_to_depth(disp_torch, 0.1, 100)
        depth_output_torch = depth_output_torch * ground_mask_torch

        # back_project
        ground_points = back_project(depth_output_torch, inv_camera_mtx_torch.unsqueeze(0)).squeeze()
        ground_points = torch.transpose(ground_points, 0, 1)
        ground_points = ground_points[torch.transpose(torch.nonzero(ground_points[:,2:3]), 0, 1)[0]]
        _, indices = torch.sort(ground_points[:,2:3].transpose(0, 1)[0])
        ground_points = ground_points[indices].squeeze()
        
        # show
        if args.show_image: 
            masked_color_depth_output_cv = monodepth_colored_output_cv * cv2.cvtColor(ground_mask_np, cv2.COLOR_GRAY2BGR)
            output_top_cv = cv2.hconcat([raw_img_cv, monodepth_colored_output_cv])
            output_buttom_cv = cv2.hconcat([bisenet_colored_cv, masked_color_depth_output_cv])
            output_cv = cv2.vconcat([output_top_cv, output_buttom_cv])
            # show_3d_points(ground_points.cpu().numpy())
            cv2.imwrite('assets/output.jpg', output_cv)
        
        ground_points = ground_points.cpu().numpy()
        total_plane = get_plane_by_lstsq(ground_points[int(len(ground_points) * 4 / 5)::len(ground_points) // 10000,])
        total_angle = get_plane2Zaxis_angle(total_plane)

        start = time.time()

        delta = (ground_points[-1,2] - ground_points[0,2]) / 10
        if delta < 0.1:
            delta = 0.1
        depth_values = np.linspace(ground_points[0,2], ground_points[-1,2], 10)
        angle_values = []
        used_flag_values = []
        for i in depth_values:
            sample_points = ground_points[(ground_points[:,2] < i + delta) & (ground_points[:,2] > i)]
            if(len(sample_points) > 400):
                sample_points = sample_points[::len(sample_points) // 400,]
            
            if(len(sample_points) > 20):
                plane = get_plane_by_ransac(sample_points, 10000, delta / 5, 0.80)
                # plane = get_plane_by_lstsq(sample_points)
                angle = get_plane2Zaxis_angle(plane)
                if angle - total_angle < 0.2:
                    angle_values.append(angle)
                    used_flag_values.append(True)
                else:
                    used_flag_values.append(False)
            else:
                used_flag_values.append(False)
        
        if len(used_flag_values) < 2:
            print("Can not get slope from image")

        depth_values = depth_values[np.array(used_flag_values)] * 3

        end = time.time()
        print("depth", depth_values)
        print("angle", angle_values)
        print("Time:", end - start)
        
        if args.show_image:
            plt.scatter(depth_values, angle_values)
            plt.ylim(0, np.pi / 6)
            plt.xlabel("depth(m)")
            plt.ylabel("angle(rad)")
            plt.show()
            plt.close() 

    print("Finish !")


if __name__ == '__main__':
    main(parse.parse_args())