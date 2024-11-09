import cv2 
import numpy as np
import sys
from ObjectDetection import ContourDetector
from screeninfo import get_monitors
from consts import Consts
from PyQt5 import QtWidgets
from windows import ProjectorWindow, UserWindow

refImages = ['images/pattern1.png', 'images/pattern2.png', 'images/pattern3.png']

def calibrate(imgIndex: int):
    try:
        # Open image
        refImg = cv2.imread(refImages[imgIndex], cv2.IMREAD_GRAYSCALE)

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
        window = ProjectorWindow(refImg)
        window.show()

        # Run Qt, exits after picture taken
        app.exec_()

        # read image taken by Qt app
        frame = cv2.imread(Consts.CALIBRATION_IMAGE_PATH)

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

        contour = ContourDetector.getMask(frame)
        ContourDetector.scaleImage(contour, dst)

        # using drawing function for the frame 
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3) 

        cv2.imwrite(Consts.HOMOGRAPHY_IMAGE_PATH, homography)
        
        # app = QtWidgets.QApplication(sys.argv)
        window = UserWindow(homography)
        window.show()

        # run Qt, exits after picture taken
        app.exec_()

    except Exception as e:
        print(e)
        cv2.destroyAllWindows()

        calibrate((imgIndex + 1) % len(refImages)) # Try the next reference image in a circular array

    finally:
    	cv2.destroyAllWindows()

def main():
    assert len(get_monitors()) > 1 # throws if no webcam connected
    calibrate(0)
    
if __name__ == '__main__':
    main()
