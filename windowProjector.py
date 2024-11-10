import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from consts import Consts
from camera import Camera
from window import Window

class ProjectorWindow(Window):
    def captureImage(self):
        cam = Camera()
        frame = cam.getFrame()

        # write to file
        cv2.imwrite(Consts.CALIBRATION_IMAGE_PATH, frame)

        # close app
        self.close()

    def __init__(self, img):
        super().__init__(Consts.PROJECTION_WINDOW_NAME)
        
        # label to hold the image
        self.label = QtWidgets.QLabel(self)
        self.setCentralWidget(self.label)
        
        # create pixmap
        self.label.setPixmap(self.imgToPixmap(img))
        
        # get screen geometry
        screen = QtWidgets.QApplication.screens()[Consts.PROJECTOR_INDEX]
        geometry = screen.geometry()
        
        # move and resize the window to fit
        self.setGeometry(geometry)
        self.showFullScreen()

        # capture image right away using a timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.captureImage)
        self.timer.start()
