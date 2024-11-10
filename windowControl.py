import cv2
import sys
import os
from consts import Consts
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLineEdit
from PyQt5.QtCore import Qt, QTimer
from window import Window
from consts import Consts
from dalle import DallE

class WindowControl(Window):
    def __init__(self, queue1, queue2, messages):
        super().__init__(Consts.CONTROL_WINDOW_NAME)
        self.queue1 = queue1
        self.queue2 = queue2
        self.messages = messages
        self.createUI()

        timer = QTimer(self)
        timer.timeout.connect(self.display_frame1)
        timer.timeout.connect(self.display_frame2)
        timer.start(10)
        

    def createUI(self):
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        screen_width, screen_height = screen_geometry.width(), screen_geometry.height()

        self.image_width = int(screen_width * 0.5)
        self.image_height = int(screen_height * 0.5)

        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        vBox = QtWidgets.QVBoxLayout(centralWidget)
        vBox.setContentsMargins(0, 0, 0, 0)
        vBox.setSpacing(10)

        self.image1 = QLabel(self)
        self.image1.setAlignment(Qt.AlignCenter)
        self.setImage1(cv2.imread(Consts.CALIBRATION_IMAGE_PATH))

        self.image1Text = QLabel("Video Preview")
        self.image1Text.setAlignment(Qt.AlignCenter)

        self.image2 = QLabel(self)
        self.image2.setAlignment(Qt.AlignCenter)
        self.setImage2(cv2.imread(Consts.HOMOGRAPHY_IMAGE_PATH))

        self.image2Text = QLabel("Sensor Feed")
        self.image2Text.setAlignment(Qt.AlignCenter)

        imageBox = QHBoxLayout()
        imageBox.setContentsMargins(0, 0, 0, 0)
        imageBox.setSpacing(20)

        image1Box = QVBoxLayout()
        image1Box.setContentsMargins(0, 0, 0, 0)
        image1Box.setSpacing(5)
        image1Box.addWidget(self.image1)
        image1Box.addWidget(self.image1Text, alignment=Qt.AlignTop)

        image2Box = QVBoxLayout()
        image2Box.setContentsMargins(0, 0, 0, 0)
        image2Box.setSpacing(5)
        image2Box.addWidget(self.image2)
        image2Box.addWidget(self.image2Text, alignment=Qt.AlignTop)

        imageBox.addLayout(image1Box)
        imageBox.addLayout(image2Box)
        vBox.addLayout(imageBox)

        buttonsBox = QHBoxLayout()
        vBox.addLayout(buttonsBox)

        textureBox = QVBoxLayout()
        buttonsBox.addLayout(textureBox)

        textureSelectBox = QHBoxLayout()
        textureBox.addLayout(textureSelectBox)

        self.textureComboBox = QComboBox()
        self.populateTextures()
        textureSelectBox.addWidget(self.textureComboBox)

        applyTextureButton = QPushButton("Apply Texture")
        applyTextureButton.setStyleSheet("font-size: 14px; padding: 8px;")
        applyTextureButton.clicked.connect(self.applyTexture)
        textureSelectBox.addWidget(applyTextureButton)
        
        genTextureButton = QPushButton("Generate Texture...")
        genTextureButton.setStyleSheet("font-size: 14px; padding: 8px;")
        genTextureButton.clicked.connect(self.generateTexture)
        textureBox.addWidget(genTextureButton)

        # calibrateButton = QPushButton("Calibrate Homography")
        # calibrateButton.setStyleSheet("font-size: 14px; padding: 8px;")
        # calibrateButton.clicked.connect(self.calibrateHomography)
        # buttonsBox.addWidget(calibrateButton)

    def display_frame1(self):
        image = self.queue1.get()
        self.setImage1(image)
    
    def display_frame2(self):
        image = self.queue2.get()
        self.setImage2(image)

    def setImage1(self, img):
        scaled_pixmap = self.imgToPixmap(img).scaled(self.image_width, self.image_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image1.setPixmap(scaled_pixmap)

    def setImage2(self, img):
        scaled_pixmap = self.imgToPixmap(img).scaled(self.image_width, self.image_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image2.setPixmap(scaled_pixmap)

    def generateTexture(self):
        print('Generate texture')

        self.w = GenerateTextureWindow(self.textureComboBox, self.populateTextures, self.messages)
        self.w.show()

    def applyTexture(self):
        self.messages.put(self.textureComboBox.currentText())

    def populateTextures(self):
        self.textureComboBox.clear()
        if os.path.exists(Consts.TEXTURES_PATH) and os.path.isdir(Consts.TEXTURES_PATH):
            file_list = [os.path.join(Consts.TEXTURES_PATH, f) for f in os.listdir(Consts.TEXTURES_PATH) if os.path.isfile(os.path.join(Consts.TEXTURES_PATH, f))]
            self.textureComboBox.addItems(file_list)
        else:
            print("Directory does not exist")

class GenerateTextureWindow(QWidget):
    def sendQuery(self):
        print('Sending query to Dall-E')
        print(self.input.text())

        self.loadingLabel = QtWidgets.QLabel(self)
        self.loadingLabel.setFixedSize(100, 100)
        
        self.movie = QtGui.QMovie(Consts.LOADING_GIF)
        self.loadingLabel.setMovie(self.movie)
        self.movie.start()

        self.vBox.addWidget(self.loadingLabel)
        QApplication.processEvents() # manual UI refresh

        client = DallE()
        fileName = client.generateImage(self.input.text())
        self.populateTextures()
        self.textureComboBox.setCurrentText(fileName)

        self.messages.put(fileName)
        # pass filename
        # call this V
        self.close()
        self.deleteLater()

    def __init__(self, textureComboBox: QComboBox, populateTextures, messages):
        super().__init__()
        self.setWindowTitle("Generate Texture")
        self.textureComboBox = textureComboBox
        self.populateTextures = populateTextures
        self.messages = messages

        screenGeometry = QApplication.primaryScreen().availableGeometry()
        screenWidth, screenHeight = screenGeometry.width(), screenGeometry.height()

        text = QLabel('Enter a prompt to generate a texture using DALL-E 3:')
        self.input = QLineEdit()
        self.input.move(int(screenWidth/2), int(screenHeight/2))
        self.input.resize(140,20)

        generate = QPushButton("Generate")
        generate.setStyleSheet("font-size: 14px; padding: 8px;")
        generate.clicked.connect(self.sendQuery)

        self.vBox = QVBoxLayout()
        self.vBox.addWidget(text, alignment=Qt.AlignBottom)

        hBox = QHBoxLayout()
        self.vBox.addLayout(hBox)
        hBox.addWidget(self.input)
        hBox.addWidget(generate)

        self.setLayout(self.vBox)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = WindowControl()

    # geometry = QApplication.primaryScreen().availableGeometry()
    # width, height = geometry.width(), geometry.height()
    # window.resize(int(width), int(height))
    window.show()

    app.exec_()

if __name__ == '__main__':
    main()
