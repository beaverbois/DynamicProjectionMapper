import cv2
import numpy
from consts import Consts
from screeninfo import get_monitors

from windows import ProjectorWindow, UserWindow
from PyQt5 import QtWidgets

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

class ContourDetector():
    # Constants
    differenceThresh = 5
    updateFreq = 20

    # Counter for tracking number of times a new frame is sent in
    numUpdates = 0

    def __init__(self, dst, homography):
        # Get values from dst
        self.dstMinX = dst[0][0][0]
        self.dstMinY = dst[0][0][1]
        self.dstMaxX = dst[0][0][0]
        self.dstMaxY = dst[0][0][1]

        for i in range(len(dst)):
            self.dstMinX = min(self.dstMinX, dst[i][0][0])
            self.dstMaxX = max(self.dstMaxX, dst[i][0][0])
            self.dstMinY = min(self.dstMinY, dst[i][0][1])
            self.dstMaxY = max(self.dstMaxY, dst[i][0][1])

        self.xDist = self.dstMaxX - self.dstMinX
        self.yDist = self.dstMaxY - self.dstMinY
        self.dst = dst

        # Initialize foreground mask to set after calibration
        self.foregroundMask = []

        # Initialize foreground identifier
        self.backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        # Save homography matrix
        self.homography = homography

        # self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        # self.model.eval()

        self.model = cv2.CascadeClassifier("images/calib/haarcascade_frontalface_default.xml")

    def processFrame(self, img):
        # Convert the image to HSV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if len(self.foregroundMask) == 0:
            self.foregroundMask = numpy.full_like(img, (255, 255, 255), dtype=numpy.uint8)
            self.foregroundMask = cv2.cvtColor(self.foregroundMask, cv2.COLOR_BGR2GRAY)

        self.last = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200, apertureSize=3)

        # Dilate and erode
        k1 = numpy.ones((1, 1), numpy.uint8)
        kernel = numpy.ones((3, 3), numpy.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, k1, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=5)
        
        # Draw in known bounds for contour identification: screen and edge of image
        edges = cv2.drawContours(edges, [self.dst], -1, (255, 255, 255), 10)
        edges = cv2.rectangle(edges, (0, 0), (edges.shape[1], edges.shape[0]), (255, 255, 255), 10)

        # Find contours from edges
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Feed the contours found back into edges to close gaps in edge detection
        for contour in contours:
            # Get the bounding box of each contour
            x, y, w, h = cv2.boundingRect(contour)

            # Ignore any contours that are too large to improve the edge detection
            if w * h > int((self.xDist * self.yDist) / 1.5):
                continue

            # Polygon bounding box approximation approach
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.05 * perimeter  # 0.05 did best in testing, 0.02 was minimum effective
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            edges = cv2.drawContours(edges, [approx_polygon], -1, (255, 255, 255), 10)

            # # Poly lines approach
            # pts = contour.reshape((-1, 1, 2))
            # edges = cv2.polylines(edges, [pts], True, (255, 255, 255), 10)
        
        # edges = cv2.drawContours(edges, [self.dst], -1, (255, 255, 255), 10)

        # Find contours a second time, using other bounding boxes to help close gaps and increase accuracy
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour, which we know to be the screen 
        maxSize = 0
        maxContour = None
        for contour in contours:
            # Get the bounding box of each contour
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out contours outside the projected image while taking the maximum-sized contour
            if w*h > maxSize and w*h < self.xDist * self.yDist and x > self.dstMinX and x+w < self.dstMaxX and y > self.dstMinY and y+h < self.dstMaxY:
                maxSize = w*h
                maxContour = contour

        # Finally, create the mask by drawing the screen contour in white over a black mat. Save in self.backgroundMask
        contour_image = numpy.zeros_like(edges, dtype=numpy.uint8)
        cv2.drawContours(contour_image, [maxContour], -1, (255, 255, 255), cv2.FILLED)

        self.backgroundMask = contour_image

        # # Look for people
        # trf = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # inp = trf(img).unsqueeze(0)
        # out = self.model(inp)['out']
        # mask = out.squeeze().argmax(0) == 15
        # masked_image = numpy.array(img) * mask.numpy()[:,:,None]
        # masked_image = Image.fromarray(masked_image.astype('uint8'), 'RGB')
        # plt.imshow(masked_image)
        # plt.axis('off')
        # plt.show()

    def __updateMask(self, frame):
        # Get the foreground mask, update model
        # fg_mask = self.backgroundSubtractor.apply(frame)

        # # Morphology for noise cleaning
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # # Invert mask: anything white is set to black, anything else is set to white
        # all = numpy.full_like(frame, (255, 255, 255), dtype=numpy.uint8)
        # fg_mask = cv2.bitwise_not(fg_mask, all)

        # # Save in self.foregroundMask
        self.foregroundMask = numpy.full_like(frame, (255, 255, 255), dtype=numpy.uint8)
        face_cor = self.model.detectMultiScale(frame)
        if len(face_cor) != 0:
            for face in face_cor:
                tmp = numpy.full_like(frame, (255, 255, 255), dtype=numpy.uint8)
                x, y, w, h = face
                x2, y2 = x+w, y+h
                tmp = cv2.rectangle(tmp, (x, y), (x2, y2), (0, 0, 0), cv2.FILLED)
                self.foregroundMask = cv2.bitwise_and(tmp, self.foregroundMask)


    def checkForChange(self, frame: cv2.Mat):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update backgroundMask (and foregroundMask) at a set interval
        self.numUpdates += 1
        if self.numUpdates >= self.updateFreq:
            self.numUpdates = 0
            # self.processFrame(frame)
            self.__updateMask(frame)
            return
        
        # Check for enough change to know there was movement. If so, update the foreground mask
        avr = numpy.mean(numpy.abs(cv2.subtract(gray.astype(numpy.int16), self.last.astype(numpy.int16))))
        if avr > self.differenceThresh:
            self.__updateMask(frame)

    def interpolateImage(self):
        # Koala
        projection = cv2.imread(Consts.CALIBRATION_IMAGES[0])
        
        # Assemble the full mask. Known background is in white on backgroundMask, known foreground is in black on foregroundMask
        mask = cv2.bitwise_and(self.backgroundMask, self.foregroundMask)

        # Use the homography matrix to map each part of the mask to the equivalent part on the projection
        mask_transform = cv2.warpPerspective(mask, numpy.linalg.inv(self.homography), (projection.shape[1], projection.shape[0]))

        # # Get the edges of the mask, can use to outline each foreground object
        # outlineMat = cv2.Canny(mask, threshold1=100, threshold2=200)
        # kernel = numpy.ones((3, 3), numpy.uint8)
        # outlineMat = cv2.morphologyEx(outlineMat, cv2.MORPH_DILATE, kernel=kernel, iterations=3)

        # tmp = numpy.full_like(outlineMat, 100, dtype=numpy.uint8)
        # tmp[outlineMat == 255] = outlineMat[outlineMat == 255]

        # Map the region of the projection we want to itself, leaving the rest as black
        contour_region = cv2.bitwise_and(projection, projection, mask=mask_transform)

        # Finally, return our fully-transformed, masked projection!
        return contour_region

    def test():
        image = cv2.imread("images/test.jpg", cv2.IMREAD_UNCHANGED)
        screen = get_monitors()[Consts.DISPLAY_INDEX]

        image = cv2.resize(image, (screen.width, screen.height))

        print(image.shape)

        cd = ContourDetector(numpy.int32([[[0, 0]], [[0, 900]], [[1200, 900]], [[1200, 0]]]), None)
        cd.processFrame(image)
        cd.processFrame(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)
        cd.checkForChange(image)

        # cv2.imshow("", cd.foregroundMask)
        cv2.imshow("Big Mask", cv2.bitwise_and(cd.foregroundMask, cd.backgroundMask))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ContourDetector.test()