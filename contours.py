import cv2
import numpy
from consts import Consts
from screeninfo import get_monitors

from windows import ProjectorWindow, UserWindow
from PyQt5 import QtWidgets
import sys

class ContourDetector():
    blurXMod = 0.005
    blurYMod = 0.005
    
    dilateXMod = 0.02
    dilateYMod = 0.01

    differenceThresh = 10
    numChangedPix = 0.001

    numUpdates = 0
    updateFreq = 20

    def __init__(self, dst, homography):
        # Get values from dst

        # print(dst)
        # dst = numpy.array(dst)

        self.homography = homography

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

        self.numChangedPix *= self.xDist * self.yDist

        self.foregroundMask = None
        self.backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        self.homography = homography

    def processFrame(self, img):
        # Convert the image to HSV
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if self.foregroundMask == None:
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
        
        edges = cv2.drawContours(edges, [self.dst], -1, (255, 255, 255), 10)

        # Show the original and edge-detected images
        # cv2.imshow('Original Image', img)
        # cv2.imshow('Canny Edge Detection', edges)

        # Find contours from the edges image
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image to draw the contours on
        contour_image = numpy.zeros_like(edges, dtype=numpy.uint8)
        # maxContour_image = image.copy()
        # flooded_image = image.copy()

        for contour in contours:
            # Get the bounding box of each contour
            x, y, w, h = cv2.boundingRect(contour)

            if w * h > int((self.xDist * self.yDist) / 1.5):
                continue

            pts = contour.reshape((-1, 1, 2))

            # cv2.rectangle(edges, (x, y), (x + w, y + h), (255, 255, 255), 2)
            perimeter = cv2.arcLength(contour, True)
        
            # Set epsilon to 2% of the perimeter (you can adjust this for more/less simplification)
            epsilon = 0.05 * perimeter  # This controls the approximation accuracy
        
            # Approximate the contour
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

            edges = cv2.drawContours(edges, [approx_polygon], -1, (255, 255, 255), 10)
            # edges = cv2.polylines(edges, [pts], True, (255, 255, 255), 10)
        
        # # edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=19)
        
        edges = cv2.drawContours(edges, [self.dst], -1, (255, 255, 255), 10)

        # cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
        # cv2.imshow("Round 1", edges)
        # cv2.waitKey(0)  
        # cv2.destroyAllWindows()

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through each contour
        maxSize = 0
        maxContour = None
        numContours = 0
        for contour in contours:
            # Get the bounding box of each contour
            x, y, w, h = cv2.boundingRect(contour)

            if w*h > maxSize and w*h < self.xDist * self.yDist and x > self.dstMinX and x+w < self.dstMaxX and y > self.dstMinY and y+h < self.dstMaxY:
                maxSize = w*h
                maxContour = contour

            numContours += 1

        # maxContour = max(contours, key=cv2.contourArea)
        
        #     # # cv2.floodFill(contour_image, None, (x - w/2, y - h/2), (255, 0, 0))

        #     # M = cv2.moments(contour)
        #     # if M["m00"] != 0:  # Avoid division by zero
        #     #     cx = int(M["m10"] / M["m00"])
        #     #     cy = int(M["m01"] / M["m00"])
        #     # else:
        #     #     cx, cy = contour[0][0]  # Fallback to the first point if the contour is degenerate

        #     # # Flood fill the contour region starting from the center
        #     # cv2.floodFill(flooded_image, None, (cx, cy), (255, 0, 0))
        
        #     

        # x, y, w, h = cv2.boundingRect(maxContour)
        
        # Draw image
        # cv2.drawContours(img, [maxContour], -1, (255, 255, 255), cv2.FILLED)
        # cv2.imshow("Contour", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.drawContours(contour_image, [maxContour], -1, (255, 255, 255), cv2.FILLED)

        # self.mask = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
        self.backgroundMask = contour_image

    def __updateMask(self, frame):
        fg_mask = self.backgroundSubtractor.apply(frame)

        # Optional: Perform morphological operations to clean up the mask (remove noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Define a kernel for morphological operation
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)  # Close small holes in the foreground mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)   # Remove small noise

        for row in range(len(fg_mask)):
            for col in range(len(fg_mask[0])):
                fg_mask[row][col] = 255 if fg_mask[row][col] < 255 else 0

        self.foregroundMask = fg_mask

    def checkForChange(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Update on a set interval
        self.numUpdates += 1
        if self.numUpdates >= self.updateFreq:
            self.numUpdates = 0
            self.processFrame(frame)
            self.__updateMask(frame)
            return
        # Check for enough change to redo the mask
        numChanged = 0
        # avrChange = numpy.mean(cv2.absdiff(self.last, gray))
        for row in range(len(gray)):
            for col in range(len(gray[0])):
                if abs(int(gray[row][col]) - int(self.last[row][col])) > self.differenceThresh: # + avrChange
                    # print("Found difference", abs(int(gray[row][col]) - int(self.last[row][col])), gray[row][col], self.last[row][col])
                    numChanged += 1
                    if numChanged > self.numChangedPix:
                        self.__updateMask(frame)
                        return

    def interpolateImage(self):
        # Koala
        projection = cv2.imread("images/pattern1.png")
        
        self.mask = cv2.bitwise_and(self.backgroundMask, self.foregroundMask)
        # cv2.imshow("", self.backgroundMask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # mask = numpy.zeros_like(projection, dtype=numpy.uint8)  # Create a blank mask
        # cv2.fillPoly(mask, [contour], (255))  # Fill the largest contour in the mask

        # # Resize the target image to match the source image size (if necessary)
        # target_resized = cv2.resize(projection, (projection.shape[0], projection.shape[1]))

        # Use the inverse homography matrix to project the mask onto the projection image
        # Convert contour to homogeneous coordinates (x, y, 1)
        # contour_homogeneous = numpy.hstack([contour.reshape(-1, 2), numpy.ones((contour.shape[0], 1))])

        # # Apply the inverse homography to each point in the contour
        # contour_transformed = []

        # for point in contour_homogeneous:
        #     # Apply inverse homography
        #     transformed_point = numpy.dot(numpy.linalg.inv(homography), point.T)
        #     transformed_point /= transformed_point[2]  # Normalize to make it homogeneous again
        #     contour_transformed.append(transformed_point[:2])  # Keep only (x, y)

        # # Convert the transformed points back to an array
        # contour_transformed = numpy.array(contour_transformed, dtype=numpy.int32)

        # # Create a binary mask where the largest contour is filled in
        # mask = numpy.zeros_like(image, dtype=numpy.uint8)  # Create a blank mask
        # cv2.fillPoly(mask, [contour_transformed], (255))  # Fill the largest contour in the mask

        # # Create a blank canvas for the output image
        # output_image = numpy.zeros((300, 300, 3), dtype=numpy.uint8)

        # # Draw the original contour in blue
        # cv2.polylines(output_image, [contour], isClosed=True, color=(255, 0, 0), thickness=2)

        # # Draw the transformed contour in green
        # cv2.polylines(output_image, [contour_transformed], isClosed=True, color=(0, 255, 0), thickness=2)

        mask_transform = cv2.warpPerspective(self.mask, numpy.linalg.inv(self.homography), (projection.shape[1], projection.shape[0])) # Set this variable

        # Extract the region inside the contour from the source image using the mask
        contour_region = cv2.bitwise_and(projection, projection, mask=mask_transform)

        # # Use the inverse mask to keep the background of the target image
        # background = cv2.bitwise_and(contour_region, contour_region, mask=inverse_mask)

        # # Overlay the contour region onto the target image
        # result = cv2.add(background, contour_region)

        # cv2.imshow("Output", contour_region)
        return contour_region

        # # cv2.imshow("Mask", mask_transform)

        # # cv2.imshow("Image with Contours", contour_image)
        # # # cv2.imshow("Image Flooded", flooded_image)

        # # Wait for a key press and close the windows
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Find connected components (regions) in the closed edge image
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(contour_image)

        # # Find the largest connected component based on the area (stats[:, cv2.CC_STAT_AREA] gives areas)
        # largest_component_index = numpy.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # Skip the background label

        # # Create a mask for the largest connected component
        # largest_component_mask = numpy.zeros_like(contour_image, dtype=numpy.uint8)
        # largest_component_mask[labels == largest_component_index] = 255

        # # Extract the region of interest (ROI) using the mask
        # region_inside = cv2.bitwise_and(image, image, mask=largest_component_mask)

        # # Display the results
        # # cv2.imshow("Canny Edges", edges)  # Display the Canny edges
        # # cv2.imshow("Morphologically Closed Edges", edges)  # Show the closed edges
        # cv2.imshow("Largest Region Inside Edges", region_inside)  # Show the region inside the largest contour

        # Wait for a key press and close the windows

    def test():
        image = cv2.imread("images/test.jpg", cv2.IMREAD_UNCHANGED)
        screen = get_monitors()[Consts.DISPLAY_INDEX]
        # screenRatio = float(screen.width) / float(screen.height)
        # dim = image.shape
        # ratio = float(dim[0]) / float(dim[1])
        # w = screen.width if ratio >= screenRatio else screen.height * ratio
        # h = screen.height if ratio <= screenRatio else screen.width / ratio

        image = cv2.resize(image, (screen.width, screen.height))

        print(image.shape)

        cd = ContourDetector(numpy.int32([[[0, 0]], [[0, 900]], [[1200, 900]], [[1200, 0]]]), None)
        cd.processFrame(image)
        cd.checkForChange(image)

        # cv2.imshow("", cd.foregroundMask)
        cv2.imshow("2", cd.interpolateImage())
        cv2.imshow("Big Mask", cv2.bitwise_and(cd.foregroundMask, cd.backgroundMask))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    ContourDetector.test()