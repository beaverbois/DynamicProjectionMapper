import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from consts import Consts
from camera import Camera, Kinect
from window import Window
import time

class ProjectorWindow(Window):
    def captureImage(self):
        frame, _ = self.cam.getFrame()

        # write to file
        cv2.imwrite(Consts.CALIBRATION_IMAGE_PATH, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # close app
        self.close()

    def __init__(self, img):
        super().__init__(Consts.PROJECTION_WINDOW_NAME)
        
        self.cam = Kinect()
        # label to hold the image
        self.label = QtWidgets.QLabel(self)
        self.setCentralWidget(self.label)
        
        # create pixmap
        # self.label.setPixmap(self.imgToPixmap(img))
        
        # get screen geometry
        screen = QtWidgets.QApplication.screens()[Consts.PROJECTOR_INDEX]
        geometry = screen.geometry()
        
        # move and resize the window to fit
        self.setGeometry(geometry)
        self.showFullScreen()

        time.sleep(2)
        rgb_frame, _ = self.cam.getFrame()
        cv2.imwrite(Consts.BLANK_IMAGE_PATH, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        self.label.setPixmap(self.imgToPixmap(img))


        # capture image right away using a timer
        self.timer = QtCore.QTimer.singleShot(3000, self.captureImage)
        # self.timer.timeout.connect(self.captureImage)
        # self.timer.start()
