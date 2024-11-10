import cv2
from consts import Consts

class Camera:
    def __init__(self):
        # create camera
        self.cam = cv2.VideoCapture(Consts.CAMERA_INDEX)

        # set resolution
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, Consts.CAMERA_WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Consts.CAMERA_HEIGHT)

    def getFrame(self):
        _, frame = self.cam.read()
        return frame