import cv2
import time

class SafeCamera:
    def __init__(self, index: int = 0):
        self.index = int(index)
        self.cap = None
        self.open()

    def open(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(self.index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

    def reopen(self):
        # alias to open()
        self.open()

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            self.open()
            time.sleep(0.1)

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        return cv2.flip(frame, 1)

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass