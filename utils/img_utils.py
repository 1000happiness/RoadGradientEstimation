import cv2
import numpy as np

def color_lane_line(img):

    # get mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[::1].ravel()
    saturation = np.sort(saturation)
    low_saturation = saturation[len(saturation) // 10 * 9]
    if low_saturation < 20:
        low_saturation = 20

    # 图片饱和度太高，无法去除车道线
    if low_saturation > 150:
        return img

    low = np.array([0, 20, 0])
    high = np.array([360, 255, 255])
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img, img, mask= mask)
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    mask = (res != 0).astype(np.uint8)
    anti_mask = (res == 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    anti_mask = (mask == 0).astype(np.uint8)

    img = img * cv2.cvtColor(anti_mask, cv2.COLOR_GRAY2BGR)
    img_flag = img > 0
    img_ravel = img.ravel()[img_flag.ravel()]
    mean_color = (int(img_ravel[::3].mean()), int(img_ravel[1::3].mean()), int(img_ravel[2::3].mean()))
    new_color = (np.full([*img.shape], mean_color) * cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)).astype(np.uint8)
    img = (img + new_color)

    return img