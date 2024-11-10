import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from consts import Consts
from camera import Camera

class ProjectorStream(QtWidgets.QMainWindow):
    def __init__(self, image, monitorIndex = Consts.PROJECTOR_INDEX):
        super().__init__()
       # label to hold the image
        self.label = QtWidgets.QLabel(self)
        self.setCentralWidget(self.label)
        
        # convert OpenCV image to QImage for PyQt
        height, width, channel = image.shape
        bytesPerLine = channel * width
        q_image = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        
        # display QImage in the label
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)
        
        # get screen geometry
        screen = QtWidgets.QApplication.screens()[monitorIndex]
        geometry = screen.geometry()
        
        # move and resize the window to fit
        self.setGeometry(geometry)
        self.showFullScreen()

class WindowProjection(QtWidgets.QMainWindow):
    def captureImage(self):
        cam = Camera()
        frame = cam.getFrame()

        # write to file
        cv2.imwrite(Consts.CALIBRATION_IMAGE_qATH, frame)

        # close app
        self.close()

    def __init__(self, image, monitorIndex = Consts.PROJECTOR_INDEX):
        super().__init__()
        
        # label to hold the image
        self.label = QtWidgets.QLabel(self)
        self.setCentralWidget(self.label)
        
        # convert OpenCV image to QImage for PyQt
        height, width = image.shape
        bytesPerLine = width
        q_image = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
        
        # display QImage in the label
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)
        
        # get screen geometry
        screen = QtWidgets.QApplication.screens()[monitorIndex]
        geometry = screen.geometry()
        
        # move and resize the window to fit
        self.setGeometry(geometry)
        self.showFullScreen()

        # capture image right away using a timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.captureImage)
        self.timer.start()
