import cv2
import threading
import time

class Camera:
    frame_buffer = None
    loop_thread = None
    loop_flag = False

    def __init__(self, cameraId=0, update_interval=0.033, pixel_size=(1280, 720)):
        super().__init__()
        self.cameraId = cameraId
        self.size = pixel_size
        self.update_interval = update_interval
        self.fram_idx = 0
        self.cap = cv2.VideoCapture(cameraId)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, pixel_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pixel_size[1])
        else:
            print("Cannot open camera")

    def __del__(self):
        self.cap.release()

    def start_camera_loop(self):
        self.loop_flag = True
        if self.loop_thread == None:
            self.loop_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.loop_thread.start()
        time.sleep(1)
        return True

    def pause_camera_loop(self):
        self.loop_flag = False

    def get_current_frame(self):
        return self.frame_buffer, self.fram_idx

    def _camera_loop(self):
        interval = self.update_interval
        while self.loop_flag :
            time_0 = time.perf_counter()
            ret, frame = self.cap.read()
            self.frame_buffer = frame
            self.fram_idx += 1

            if not ret:
                self.loop_flag = False
                break

            time_1 = time.perf_counter()
            if time_1 - time_0 < interval:
                time.sleep(self.update_interval - (time_1 - time_0))