from utils.monodepth_utils import disp_to_depth
import numpy as np
import cv2

# 视差图计算
def get_depth_map(imgL, imgR, sigma=1.3):
    imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)

    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = np.abs(filteredImg)
    # filteredImg = filteredImg[:,100:]
    # filteredImg = cv2.resize(filteredImg, (1280,720))
    filteredImg[filteredImg == 0] = 1
    
    filteredImg = 52.9 * 120 / filteredImg
    # filteredImg = filteredImg / 1500
    print(filteredImg.shape)
    line = filteredImg.copy().ravel()
    line.sort()
    vmax = line[int(len(line) * 0.50)]
    # vmin = line[int(len(line) * 0.05)]
    filteredImg[filteredImg > vmax] = vmax
    # filteredImg[filteredImg < vmin] = vmin
    print(filteredImg.max(), filteredImg.min())
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg

def monodepth2_disp(imgL):
    import json
    import os
    import torch
    from PIL import Image
    from torchvision import transforms
    from nets.monodepth_v2 import ResnetEncoder, DepthDecoder
    import matplotlib as mpl
    import matplotlib.cm as cm

    device = torch.device("cuda")

    with open("config/monodepth_v2.json") as f:
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

    raw_img_cv = imgL

    original_width, original_height  = 1280, 720

    with torch.no_grad():
        # raw_img(cv) to monodepth_input(torch)
        raw_img_pil = Image.fromarray(raw_img_cv)
        resized_image_pil = raw_img_pil.resize((feed_width, feed_height), Image.LANCZOS)
        monodepth_input_torch = transforms.ToTensor()(resized_image_pil).unsqueeze(0).to(device)
        # monodepth process
        features = monodepth_encoder(monodepth_input_torch)
        disp_torch = monodepth_decoder(features)[("disp", 0)]

        # monodepth_output(torch) to colored_img(cv)
        resized_disp_torch = torch.nn.functional.interpolate(disp_torch, (original_height, original_width), mode="bilinear", align_corners=False)
        disp_resized_np = resized_disp_torch.squeeze().cpu().numpy()

        disp_resized_np = 1 / disp_resized_np
        vmax = np.percentile(disp_resized_np, 95)
        vmin = np.percentile(disp_resized_np, 5)
        disp_resized_np[disp_resized_np > vmax] = vmax
        disp_resized_np[disp_resized_np < vmin] = vmin

        print(disp_resized_np.max(), disp_resized_np.min())
        filteredImg = cv2.normalize(disp_resized_np,  disp_resized_np, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        
        # disp_resized_np = 1 / disp_resized_np
        # print(disp_resized_np.max(), disp_resized_np.min())
        # vmax = np.percentile(disp_resized_np, 95)
        # vmin = np.percentile(disp_resized_np, 5)
        # normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        # monodepth_colored_output = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        # monodepth_colored_output_cv = cv2.cvtColor(monodepth_colored_output, cv2.COLOR_RGB2GRAY)
        # return monodepth_colored_output_cv
        return filteredImg

if __name__ == "__main__":
    # imgL = cv2.imread('assets/opencv/ambush_5_left.jpg', 0)
    # imgR = cv2.imread('assets/opencv/ambush_5_right.jpg', 0)

    # imgL = cv2.imread('assets/kitti/1_l.jpg', 0)
    # imgR = cv2.imread('assets/kitti/1_r.jpg', 0)

    # imgL = cv2.imread('assets/zed1_test/14594_l.jpg')
    # imgR = cv2.imread('assets/zed1_test/14594_r.jpg')

    imgL = cv2.imread('svo_data/zed1/17202_l.jpg')
    imgR = cv2.imread('svo_data/zed1/17202_r.jpg')
    
    # imgL = cv2.imread('assets/zed1_test/0_l.jpg')
    # imgR = cv2.imread('assets/zed1_test/0_r.jpg')

    disp1 = get_depth_map(imgL, imgR)
    disp2 = monodepth2_disp(imgL)

    h1 = cv2.hconcat([disp1, disp2])
    h2 = cv2.hconcat([imgL, imgR])
    h1 = cv2.cvtColor(h1, cv2.COLOR_GRAY2RGB)
    output = cv2.vconcat([h1, h2])

    cv2.imwrite('disparity.jpg', output)

