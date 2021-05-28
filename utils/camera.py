import cv2
import threading
import time
import os
import pyzed.sl as sl

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

        if type(cameraId) == str:
            self.svo_flag = cameraId.split('.')[1] == 'svo'
        else:
            self.svo_flag = False

    def start_camera_loop(self):
        if self.svo_flag:
            input_type = sl.InputType()
            input_type.set_from_svo_file(self.cameraId)
            self.init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
            self.cam = sl.Camera()
            status = self.cam.open(self.init)
            if status != sl.ERROR_CODE.SUCCESS:
                print("Cannot open camera")
                return False
            self.runtime = sl.RuntimeParameters()
            self.mat = sl.Mat()
        else:
            self.cap = cv2.VideoCapture(self.cameraId)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            else:
                print("Cannot open camera")
                return False

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
        while self.loop_flag:
            time_0 = time.perf_counter()
            if self.svo_flag:
                ret = self.cam.grab(self.runtime)
                if ret == sl.ERROR_CODE.SUCCESS:
                    self.cam.retrieve_image(self.mat, view=sl.VIEW.LEFT)
                    self.frame_buffer = cv2.cvtColor(self.mat.get_data(), cv2.COLOR_RGBA2RGB)     
                    self.fram_idx += 1
                else:
                    self.loop_flag = False
                    time.sleep(5)
                    self.cam.close()
                    break
            else:
                ret, frame = self.cap.read()
                self.frame_buffer = frame
                self.fram_idx += 1
                if not ret:
                    self.loop_flag = False
                    time.sleep(5)
                    self.cap.release()
                    break

            time_1 = time.perf_counter()
            if time_1 - time_0 < interval:
                time.sleep(self.update_interval - (time_1 - time_0))

        