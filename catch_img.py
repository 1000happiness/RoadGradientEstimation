import cv2
import argparse
import os

parse = argparse.ArgumentParser()
parse.add_argument('--vedio_path', type=str, default='assets/zed1.mkv')
parse.add_argument('--img_path', type=str)

args = parse.parse_args()

cap = cv2.VideoCapture(args.vedio_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

i = 0

if not os.path.exists(args.img_path):
    os.mkdir(args.img_path)

while True:
    ret, frame = cap.read()
    # frame = cv2.flip(frame,1)
    cv2.imshow("video", frame)
    key = cv2.waitKey(50)
    if key  == ord('q'):  #判断是哪一个键按下
        break
    if key == ord('s'):
        file_name = os.path.join(args.img_path, '{}.jpg'.format(i))
        cv2.imwrite(file_name, frame)
        print("save " + file_name)
        i += 1
cv2.destroyAllWindows()