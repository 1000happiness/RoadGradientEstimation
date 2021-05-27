import cv2
import numpy as np
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--shape', type=int, nargs='+', default=[720, 1280])
parse.add_argument('--box', type=int, nargs='+', default=[0, 0, 720, 1280])
parse.add_argument('--mask_path', type=str, default='assets/mask/mask.jpg')

def create_rect_mask(args):
    shape = args.shape
    mask_np = np.zeros(shape, dtype=np.uint8)
    
    box = args.box
    mask_np[: box[0], :] = 1
    mask_np[:, : box[1]] = 1
    mask_np[box[2]: , :] = 1
    mask_np[:, box[3]: ] = 1

    mask_cv = cv2.cvtColor(mask_np * 255, cv2.COLOR_GRAY2RGBA)
    cv2.imwrite(args.mask_path, mask_cv)

if __name__ == '__main__':
    create_rect_mask(parse.parse_args())