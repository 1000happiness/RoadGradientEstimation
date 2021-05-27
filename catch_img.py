import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

i = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    cv2.imshow("video", frame)
    key = cv2.waitKey(50)
    if key  == ord('q'):  #判断是哪一个键按下
        break
    if key == ord('s'):
        cv2.imwrite('assets/calibration_data/{}.jpg'.format(i), frame)
        i += 1
cv2.destroyAllWindows()