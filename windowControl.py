import cv2
import sys
from consts import Consts
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QColor, QDoubleValidator, QIntValidator
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject, QRect
from PyQt5 import QtWidgets, QtCore, QtGui
from window import Window
from consts import Consts

class WindowControl(Window):
    def __init__(self):
        super().__init__(Consts.CONTROL_WINDOW_NAME)

        self.createUI()

    def createUI(self):
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        screen_width, screen_height = screen_geometry.width(), screen_geometry.height()

        self.image_width = int(screen_width * 0.4)
        self.image_height = int(screen_height * 0.4)

        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        rootVBox = QtWidgets.QVBoxLayout(centralWidget)
        self.setLayout(rootVBox)

        self.image1 = QLabel(self)
        self.image1.setAlignment(Qt.AlignCenter)
        self.setImage1(cv2.imread(Consts.CALIBRATION_IMAGE_PATH))

        self.image1Text = QLabel("Composite")
        self.image1Text.setAlignment(Qt.AlignCenter)

        self.image2 = QLabel(self)
        self.image2.setAlignment(Qt.AlignCenter)
        self.setImage2(cv2.imread(Consts.HOMOGRAPHY_IMAGE_PATH))
        
        self.image2Text = QLabel("Homography")
        self.image2Text.setAlignment(Qt.AlignCenter)

        imageBox = QtWidgets.QHBoxLayout()
        image1Box = QtWidgets.QVBoxLayout()
        image2Box = QtWidgets.QVBoxLayout()

        rootVBox.addLayout(imageBox)
        imageBox.addLayout(image1Box)
        imageBox.addLayout(image2Box)
        image1Box.addWidget(self.image1)
        image1Box.addWidget(self.image1Text)
        image2Box.addWidget(self.image2)
        image2Box.addWidget(self.image2Text)

        button = QPushButton("Calibrate Homography")
        button.setStyleSheet("font-size: 14px; padding: 8px;")  # Optional styling for the button
        image2Box.addWidget(button)

    def setImage1(self, img):
        scaled_pixmap = self.imgToPixmap(img).scaled(self.image_width, self.image_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image1.setPixmap(scaled_pixmap)
    
    def setImage2(self, img):
        scaled_pixmap = self.imgToPixmap(img).scaled(self.image_width, self.image_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image2.setPixmap(scaled_pixmap)

app = QtWidgets.QApplication(sys.argv)
window = WindowControl()
window.resize(600, 600)
window.show()

app.exec_()
