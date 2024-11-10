import cv2 
import numpy as np
import sys
from contours import ContourDetector
from screeninfo import get_monitors
from consts import Consts
from PyQt5 import QtWidgets
from stream import ProjectorStream
from windowProjector import ProjectorWindow
from camera import Camera, Kinect
import multiprocessing as mp
from queue import Empty
import os

import time
from progressbar import progressbar

cd = None
homography = None

def projectCalibration(refImg):
    # create Qt app and window
    app = QtWidgets.QApplication(sys.argv)
    window = ProjectorWindow(refImg)
    window.show()

    # Run Qt, exits after picture taken
    sys.exit(app.exec_())


def calibrate(imgIndex: int):
    global cd
    global homography
    try:
        # Open image
        refImg = cv2.imread(Consts.CALIBRATION_IMAGES[imgIndex], cv2.IMREAD_GRAYSCALE)

        # creating the SIFT algorithm 
        sift = cv2.SIFT_create() 

        # find the keypoints and descriptors with SIFT 
        kpImage, descImage = sift.detectAndCompute(refImg, None) 

        # initializing the dictionary 
        indexParams = dict(algorithm = 0, trees = 5) 
        searchParams = dict()

        # by using Flann Matcher
        flann = cv2.FlannBasedMatcher(indexParams, searchParams)
        
        p = mp.Process(target=projectCalibration, args=(refImg,))
        p.start()
        p.join()


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
        homography, status = cv2.findHomography(queryPts, trainPts, cv2.RANSAC, 5.0) 

        # ravel function returns 
        # contiguous flattened array 
        # matches_mask = mask.ravel().tolist() 

        # initializing height and width of the image 
        h, w = refImg.shape[:2]

        # saving all points in pts 
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2) 

        print(pts)

        # applying perspective algorithm 
        dst = cv2.perspectiveTransform(pts, homography)
        print(np.int32(dst))

        # cam = Camera()
        # time.sleep(10)
        # cam = Kinect()
        # time.sleep(10)
        # frame, _ = cam.getFrame()
        # print(frame.shape)
        n_frame = cv2.imread(Consts.CALIBRATION_IMAGE_PATH)
        print(n_frame.shape)

        # using drawing function for the frame 
        homographyImg = cv2.polylines(n_frame, [np.int32(dst)], True, (255, 0, 0), 3) 

        cv2.imshow("Homo", homographyImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # write homography image
        cv2.imwrite(Consts.HOMOGRAPHY_IMAGE_PATH, homographyImg)
        
        # identify countours
        # cd = ContourDetector(np.int32(dst), homography)
        # cd.processFrame(frame)
        # cd.interpolateImage()

        # # ---- BENCHMARK ----
        # camera = Camera()
        # frame = camera.getFrame()
        # t0 = time.time()
        # for i in progressbar(range(1000)):
        #     cd.processFrame(frame)
        #     cd.interpolateImage()
        # t1 = time.time()

        # print(f"time: {t1-t0}s | fps: {1000/(t1-t0)}")
        # # ---- BENCHMARK ----

        return cd

        # app = QtWidgets.QApplication(sys.argv)

        # run Qt, exits after picture taken
        
        # # app = QtWidgets.QApplication(sys.argv)
        # window = UserWindow(homographyImg)
        # window.show()

        # # run Qt, exits after picture taken
        # app.exec_()

    except Exception as e:
        print(e)
        cv2.destroyAllWindows()
        exit()

        calibrate((imgIndex + 1) % len(Consts.CALIBRATION_IMAGES)) # Try the next reference image in a circular array

    finally:
    	cv2.destroyAllWindows()

def videoPlayer(queue):
    # while True:
    #     try:
    #         queue.get(True, 2)
    #         # print("recieved!")
    #         queue.task_done()
    #     except Empty:
    #         print("timed out.. exiting")
    #         return
    print("Player?")

    app = QtWidgets.QApplication(sys.argv)
    window = ProjectorStream(queue)
    window.show()
    app.exec_()
    print("Player started")

def frameCreator(queue, cd, cam):
    # camera = Camera()
    print("frameCreator started")
    t0 = time.time()
    for i in range(200):
        frame, _ = cam.getFrame()
        # cd.processFrame(frame)
        # image = cd.interpolateImage()
        image = cd.thresholdFrame(frame)
        queue.put(image)

        # images = ['images/pattern1.png', 'images/pattern2.png', 'images/pattern3.png']
        # # for i in range(120):
        # image_path = images[i%3]
        # image = cv2.imread(image_path)
        # queue.put(image)
    t1 = time.time()
    print(f"fps: {1000/(t1-t0)}")
    queue.close()
    print("frameCreator Done!")


def main():
    assert len(get_monitors()) > 1 # throws if no projector connected
    cd = calibrate(0)
    cam = Kinect()

    print("calibrate done!")
    queue = mp.JoinableQueue(5)

    player = mp.Process(target=videoPlayer, args=(queue,))
    player.start()

    frameCreator(queue, cd, cam)

    player.join()
    print("joined?")
    queue.join()
    
if __name__ == '__main__':
    main()