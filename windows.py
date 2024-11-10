from queue import Empty
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from consts import Consts
from camera import Camera
import sys

class ProjectorStream(QtWidgets.QMainWindow):
    def __init__(self, queue, monitorIndex = Consts.PROJECTOR_INDEX):
        super().__init__()

        self.setWindowTitle("Projector Stream")
        screen = QtWidgets.QApplication.screens()[monitorIndex]
        geometry = screen.geometry()
        self.setGeometry(geometry)
        self.showFullScreen()
        self.queue = queue

        self.label = QtWidgets.QLabel(self)
        self.setCentralWidget(self.label)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.display_frame)
        timer.start(1000//60)

    def display_frame(self):

        try:
            image = self.queue.get(True, 10)
               
            # convert OpenCV image to QImage for PyQt
            height, width, channel = image.shape
            bytesPerLine = channel * width
            q_image = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            
            # display QImage in the label
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.label.setPixmap(pixmap)
            self.update()

            self.queue.task_done()

        except Empty:
            print("timed out.. exiting")
            self.close()


class ProjectorWindow(QtWidgets.QMainWindow):
    def captureImage(self):
        cam = Camera()
        frame = cam.getFrame()

        # write to file
        cv2.imwrite(Consts.CALIBRATION_IMAGE_PATH, frame)

        # close app
        self.close()
        print("dead")

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

        QtCore.QTimer.singleShot(500, self.captureImage)


class UserWindow(QtWidgets.QMainWindow):
    def __init__(self, image, monitorIndex = Consts.DISPLAY_INDEX):
        super().__init__()
        
        # label to hold the image
        self.label = QtWidgets.QLabel(self)
        self.setCentralWidget(self.label)

        # convert from OpenCV BGR to QImage RGB
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # convert OpenCV image to QImage
        height, width = imageRGB.shape[:2]
        bytesPerLine = width * 3 # 3 bytes per pixel (RGB)
        qImage = QtGui.QImage(imageRGB.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

        # scale QImage to fit the window size
        scaledImage = qImage.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        
        # display QImage in the label
        pixmap = QtGui.QPixmap.fromImage(scaledImage)

        # display pixmap in label
        self.label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    refImg = cv2.imread('images/pattern1.png', cv2.IMREAD_GRAYSCALE)
    window = ProjectorWindow(refImg)
    window.show()

    # Run Qt, exits after picture taken
    sys.exit(app.exec_())


