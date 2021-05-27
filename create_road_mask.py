import argparse
import json
import cv2
import glob
import os
import numpy as np
from tqdm import tqdm
import torch

from nets.bisenet_v2 import BiSeNetV2
from nets.monodepth_v2 import ResnetEncoder, DepthDecoder
from utils.monodepth_utils import disp_to_depth
from utils.bisenet_utils import ToTensor
from utils.reprojection import reprojection_to_3D
from utils.plane_utils import get_plane_by_lstsq, get_plane_by_ransac, get_plane2Zaxis_angle
from utils.plot_utils import show_3d_points
from utils.img_utils import color_lane_line

parse = argparse.ArgumentParser()
parse.add_argument('--img_folder', type=str, default='./svo_data/zed1')

parse.add_argument('--no_cuda', type=int, default=0)
parse.add_argument('--bisenet_config_path', type=str, default='config/bisenet_v2.json')

def main(args):
    # cuda or cpu
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # init bisenet_v2 network
    print('Start initializing bisenet_v2 network')
    with open(args.bisenet_config_path) as f:
        bisenet_config = json.load(f)
    bisenet = BiSeNetV2(bisenet_config['n_classes'])
    bisenet.load_state_dict(torch.load(bisenet_config['model_path'], map_location='cpu'))
    bisenet.to(device)
    bisenet.eval()

    ground_class_number = bisenet_config['ground_class_number']
    to_tensor = ToTensor(
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )
    print('Finish initializing bisenet_v2 network')

    original_width, original_height = 1280, 720

    img_list = glob.glob(os.path.join(args.img_folder, "*.jpg"))

    print('Begin !')
    with torch.no_grad():
        for img_filename in tqdm(img_list):
            raw_img_cv = cv2.imread(img_filename)
            # raw_img to bisenet input
            resized_img_cv = cv2.resize(raw_img_cv, (1024, 512))
            bisenet_input_torch = to_tensor(dict(im=resized_img_cv, lb=None))['im'].unsqueeze(0).to(device)
            
            # bisenet process
            bisenet_output_torch = bisenet(bisenet_input_torch)[0].argmax(dim=1).squeeze()

            # bisenet_output(torch) to ground_mask(np)
            ground_mask_output_np = (bisenet_output_torch.cpu().numpy() == ground_class_number).astype(np.uint8)
            ground_mask_output_np = cv2.resize(ground_mask_output_np, (original_width, original_height))

            # ground_mask(np) to bisenet_colored_output(cv)
            bisenet_colored_output_cv = cv2.cvtColor(ground_mask_output_np * 255, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(img_filename.split(".")[0] + "_road_mask.jpg", bisenet_colored_output_cv)

    print("Finish !")

if __name__ == '__main__':
    main(parse.parse_args())