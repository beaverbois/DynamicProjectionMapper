import cv2 
import numpy as np
import sys
import time
from screeninfo import get_monitors
from PyQt5 import QtWidgets, QtGui, QtCore

monitors = get_monitors()
user = monitors[0]
assert len(monitors) > 1
projectorIndex = 1
refImages = ['images/pattern1.png', 'images/pattern2.png', 'images/pattern3.png']

controlWindowName = 'Projection Control'
projectionWindowName = 'Projector'
calibrationImageName = 'calibration.jpg'

class FullScreenWindow(QtWidgets.QMainWindow):
    def captureImage(self):
        # initializing web cam  
        cam = cv2.VideoCapture(0)

        # take a picture
        _, frame = cam.read()

        # write to file
        cv2.imwrite(calibrationImageName, frame)

        # close app
        self.close()

    def __init__(self, image, monitor_index):
        super().__init__()
        
        # set up label to hold the image
        self.label = QtWidgets.QLabel(self)
        self.setCentralWidget(self.label)
        
        # convert OpenCV image to QImage for PyQt
        height, width = image.shape
        bytes_per_line = width
        q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
        
        # display QImage in the label
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)
        
        # get screen geometry
        screen = QtWidgets.QApplication.screens()[monitor_index]
        geometry = screen.geometry()
        
        # move and resize the window to fit
        self.setGeometry(geometry)
        self.showFullScreen()

        # capture image right away using a timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.captureImage)
        self.timer.start()

def calibrate(imgIndex: int):
    try:
        # Open image
        refImg = cv2.imread(refImages[imgIndex], cv2.IMREAD_GRAYSCALE)
        
        # Setup projector window
        cv2.namedWindow(controlWindowName, cv2.WINDOW_FULLSCREEN)    
        cv2.moveWindow(controlWindowName, user.x, 0)
        cv2.setWindowProperty(controlWindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # creating the SIFT algorithm 
        sift = cv2.SIFT_create() 

        # find the keypoints and descriptors with SIFT 
        kpImage, descImage = sift.detectAndCompute(refImg, None) 

        # initializing the dictionary 
        indexParams = dict(algorithm = 0, trees = 5) 
        searchParams = dict() 

        # by using Flann Matcher 
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)

        # create Qt app and window
        app = QtWidgets.QApplication(sys.argv)
        window = FullScreenWindow(refImg, projectorIndex)
        window.show()

        # Run Qt, exits after picture taken
        app.exec_()

        # read image taken by Qt app
        frame = cv2.imread(calibrationImageName)

        # converting the frame into grayscale 
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        # find the keypoints and descriptors with SIFT 
        kpGrayFrame, descGrayFrame = sift.detectAndCompute(grayFrame, None) 

        # finding nearest match with KNN algorithm 
        matches= flann.knnMatch(descImage, descGrayFrame, k=2) 

        # initialize list to keep track of only good points 
        goodPoints=[]

        for m, n in matches: 
            # append the points according 
            # to distance of descriptors 
            if(m.distance < 0.6 * n.distance): 
                goodPoints.append(m) 

        # maintaining list of index of descriptors 
        # in query descriptors 
        queryPts = np.float32([kpImage[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2) 

        # maintaining list of index of descriptors 
        # in train descriptors 
        trainPts = np.float32([kpGrayFrame[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2) 

        # finding perspective transformation 
        # between two planes 
        matrix, mask = cv2.findHomography(queryPts, trainPts, cv2.RANSAC, 5.0) 

        # ravel function returns 
        # contiguous flattened array 
        # matches_mask = mask.ravel().tolist() 

        # initializing height and width of the image 
        h, w = refImg.shape[:2]

        # saving all points in pts 
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2) 

        # applying perspective algorithm 
        dst = cv2.perspectiveTransform(pts, matrix) 

        # using drawing function for the frame 
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3) 
        
        # print(homography)

        # showing the final output 
        # with homography 
        cv2.imshow(controlWindowName, homography)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)
        cv2.destroyAllWindows()

        calibrate((imgIndex + 1) % len(refImages)) # Try the next reference image in a circular array

    finally:
    	cv2.destroyAllWindows()

def main():
    calibrate(0)
    
if __name__ == '__main__':
    main()
