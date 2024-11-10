import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from screeninfo import get_monitors
from consts import Consts

class Window(QtWidgets.QMainWindow):
    def __init__(self, title):
        super().__init__()
        self.monitor = get_monitors()[Consts.DISPLAY_INDEX]
        self.setWindowTitle(title)
        

    def imgToPixmap(self, img):
        # Convert OpenCV BGR to QImage RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert OpenCV image to QImage
        height, width = imgRGB.shape[:2]
        bytesPerLine = width * 3  # 3 bytes per pixel (RGB)
        qImage = QtGui.QImage(imgRGB.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

        return QtGui.QPixmap.fromImage(qImage)