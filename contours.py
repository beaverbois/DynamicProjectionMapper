import cv2
import numpy
from consts import Consts
from screeninfo import get_monitors

from PyQt5 import QtWidgets

class ContourDetector():
    # Scalar to account for slight differences in distance at a wall
    threshScale = 0.95

    def __init__(self, dst, homography):
        # Get values from dst for use in processFrame
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

        # Save homography matrix
        self.homography = homography

        # self.model = cv2.CascadeClassifier("images/calib/haarcascade_frontalface_default.xml")

        image = cv2.imread("images/textures/jellyfish.jpg")
        self.updateProjection(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Call every frame
    def maskImage(self, depth: numpy.array):
        # Mask all objects that are closer than a threshold
        depth = depth[:-2]
        depth[depth == float('inf')] = 0
        depth[depth > 2250] = 0
        depth[depth < 10] = 0
        gray = cv2.cvtColor(self.project, cv2.COLOR_BGR2GRAY)
        wall = numpy.bitwise_and(depth.astype(numpy.uint16), depth.astype(numpy.uint16), self.backgroundMask)
        wallTransform = cv2.warpPerspective(wall.astype(numpy.uint16), numpy.linalg.inv(self.homography), (gray.shape[1], gray.shape[0])) # Might(?) need a different shape
        _, threshArr = cv2.threshold(wallTransform, 1, 2000, cv2.THRESH_BINARY_INV)
        # threshArr = threshArr[wallTransform < numpy.inf]
        #threshArr = threshArr[wallTransform < 2000]

        #wallThresh = numpy.mean(threshArr) * self.threshScale # Not sure if this is 100% right
        #, wallMask = cv2.threshold(wallTransform, wallThresh, 255, cv2.THRESH_BINARY) # Might want to use wall here

        kernel = numpy.ones((9, 9), numpy.uint8)
        threshArr = cv2.morphologyEx(threshArr, cv2.MORPH_DILATE, kernel, 7)

        maskedImage = cv2.bitwise_and(self.project, self.project, mask=threshArr.astype(numpy.uint8))

        return maskedImage

    def updateProjection(self, image):
        # Get the dimensions of the original image
        h, w, _ = image.shape

        # Determine how many tiles to fill each dimension
        numHorizontal = (1920 // w) + 1  # +1 to ensure full coverage
        numVertical = (1080 // h) + 1  # +1 to ensure full coverage

        # Create a larger, repeating-pattern image using numpy.tile
        tiledImg = numpy.tile(image, (numVertical, numVertical, 1))
        tiledImg = cv2.resize(tiledImg, (1920, 1080))

        # Crop the image to the exact target size
        tiledImg = tiledImg[:1080, :1920]

        self.project = tiledImg

    # def thresholdFrame(self, img):
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     proj = cv2.drawContours(gray, [self.dst], -1, 255, thickness=cv2.FILLED)
    #     tmp = numpy.full_like(gray, 255, dtype=numpy.uint8)
    #     mask = cv2.bitwise_not(tmp, proj)
    #     gray[mask==255] = mask[mask==255]
    #     transform = cv2.warpPerspective(gray, numpy.linalg.inv(self.homography), (self.project.shape[1], self.project.shape[0]))
    #     diff = numpy.abs(cv2.subtract(transform.astype(numpy.int16), self.project.astype(numpy.int16)))
    #     _, thresh = cv2.threshold(diff, self.differenceThresh, 255, cv2.THRESH_BINARY_INV)
    #     return cv2.bitwise_and(self.project, self.project, thresh)

    # Use for calibration
    def processFrame(self, img):
        # Convert the image to HSV for processing
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Blur and Canny
        blurred = cv2.GaussianBlur(hsv, (3, 3), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=200, apertureSize=3)

        # Dilate and erode
        k1 = numpy.ones((1, 1), numpy.uint8)
        kernel = numpy.ones((3, 3), numpy.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, k1, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=3)
        
        # Draw in known bounds for contour identification: screen and image edges
        edges = cv2.drawContours(edges, [self.dst], -1, (255, 255, 255), 5)
        edges = cv2.rectangle(edges, (0, 0), (edges.shape[1], edges.shape[0]), (255, 255, 255), 5)

        # Find contours
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
            edges = cv2.drawContours(edges, [approx_polygon], -1, (255, 255, 255), 5)

            # # Poly lines approach
            # pts = contour.reshape((-1, 1, 2))
            # edges = cv2.polylines(edges, [pts], True, (255, 255, 255), 10)

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
        print(contour_image.shape)

        # # Look for people using an ML model
        # trf = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # inp = trf(img).unsqueeze(0)
        # out = self.model(inp)['out']
        # mask = out.squeeze().argmax(0) == 15
        # masked_image = numpy.array(img) * mask.numpy()[:,:,None]
        # masked_image = Image.fromarray(masked_image.astype('uint8'), 'RGB')
        # plt.imshow(masked_image)
        # plt.axis('off')
        # plt.show()

    # def __updateMask(self, frame):
    #     # Get the foreground mask, update model
    #     fg_mask = self.backgroundSubtractor.apply(frame)

    #     # Morphology for noise cleaning
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    #     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    #     # Invert mask: anything white is set to black, anything else is set to white
    #     all = numpy.full_like(frame, (255, 255, 255), dtype=numpy.uint8)
    #     fg_mask = cv2.bitwise_not(fg_mask, all)

    #     self.foregroundMask = fg_mask

    #     # # # Save in self.foregroundMask
    #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     # self.foregroundMask = numpy.full_like(gray, 255, dtype=numpy.uint8)
    #     # face_cor = self.model.detectMultiScale(frame)
    #     # if len(face_cor) != 0:
    #     #     for face in face_cor:
    #     #         tmp = numpy.full_like(gray, 255, dtype=numpy.uint8)
    #     #         x, y, w, h = face
    #     #         x2, y2 = x+w, y+h
    #     #         tmp = cv2.rectangle(tmp, (max(0, x-w//2), max(0, y-h//2)), (min(tmp.shape[1], x2+w//2), min(tmp.shape[0], y2+h//2)), 0, cv2.FILLED)
    #     #         # tmp = cv2.ellipse(tmp, (x+w/2, y+h/2), (w/2,h/2), 0, 0, 0, color=0, thickness=cv2.FILLED)
    #     #         self.foregroundMask = cv2.bitwise_and(tmp, self.foregroundMask)

    #     # self.last = frame

    # def checkForChange(self, frame: cv2.Mat):
    #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     # Update backgroundMask (and foregroundMask) at a set interval
    #     # self.numUpdates += 1
    #     # if self.numUpdates >= self.updateFreq:
    #     #     self.numUpdates = 0
    #     #     # self.processFrame(frame)
    #     #     self.__updateMask(frame)
    #     #     return
        
    #     # Check for enough change to know there was movement. If so, update the foreground mask
    #     # avr = numpy.mean(numpy.abs(cv2.subtract(gray.astype(numpy.int16), self.last.astype(numpy.int16))))
    #     # if avr > self.differenceThresh:
    #     #     self.__updateMask(frame)
    #     self.__updateMask(frame)
    #     # self.processFrame(frame)

    # def interpolateImage(self):
    #     # Koala
    #     projection = self.project
        
    #     # Assemble the full mask. Known background is in white on backgroundMask, known foreground is in black on foregroundMask
    #     mask = cv2.bitwise_and(self.backgroundMask, self.foregroundMask)

    #     cv2.imshow("m", mask)

    #     # Use the homography matrix to map each part of the mask to the equivalent part on the projection
    #     mask_transform = cv2.warpPerspective(mask, numpy.linalg.inv(self.homography), (self.project.shape[1], self.project.shape[0]))

    #     # # Get the edges of the mask, can use to outline each foreground object
    #     # outlineMat = cv2.Canny(mask, threshold1=100, threshold2=200)
    #     # kernel = numpy.ones((3, 3), numpy.uint8)
    #     # outlineMat = cv2.morphologyEx(outlineMat, cv2.MORPH_DILATE, kernel=kernel, iterations=3)

    #     # tmp = numpy.full_like(outlineMat, 100, dtype=numpy.uint8)
    #     # tmp[outlineMat == 255] = outlineMat[outlineMat == 255]

    #     # Map the region of the projection we want to itself, leaving the rest as black
    #     contour_region = cv2.bitwise_and(projection, projection, mask=mask_transform)

    #     self.last = cv2.cvtColor(contour_region, cv2.COLOR_BGR2GRAY)

    #     # Finally, return our fully-transformed, masked projection!
    #     return contour_region

    def test():
        image = cv2.imread("images/test.jpg", cv2.IMREAD_UNCHANGED)
        screen = get_monitors()[Consts.DISPLAY_INDEX]

        image = cv2.resize(image, (screen.width, screen.height))

        cd = ContourDetector(numpy.int32([[[0, 0]], [[0, 900]], [[1200, 900]], [[1200, 0]]]), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # cv2.imshow("Img", cd.project)
        cd.processFrame(image)
        cd.maskImage(numpy.zeros(1200, 1920))
        # cd.processFrame(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.checkForChange(image)
        # cd.thresholdFrame(image)

if __name__ == '__main__':
    ContourDetector.test()