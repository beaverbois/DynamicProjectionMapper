import cv2
import numpy
import dalle
from consts import Consts
from screeninfo import get_monitors

from PyQt5 import QtWidgets

class ContourDetector():
    # # Constants
    # differenceThresh = 50
    # updateFreq = 20

    # # Counter for tracking number of times a new frame is sent in
    # numUpdates = 0

    threshScale = 0.90

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

        # # Initialize foreground mask to set after calibration
        # self.foregroundMask = []

        # # Initialize foreground identifier
        # self.backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        # Save homography matrix
        self.homography = homography

        # self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        # self.model.eval()

        # self.model = cv2.CascadeClassifier("images/calib/haarcascade_frontalface_default.xml")

        image = cv2.imread("images/textures/taco.jpg")
        self.updateProjection(image)

    # Call every frame
    def maskImage(self, depth):
        # Mask all objects that are closer than a threshold
        gray = cv2.cvtColor(self.project, cv2.COLOR_BGR2GRAY)
        # for i in range(len(depth)):
        #     for j in range(len(depth[0])):
        #         print("Depth", i, j, depth[i][j])
        # cv2.imshow("De", depth.astype(numpy.uint16))
        # for i in range(len(depth)):
        #     for j in range(len(depth[0])):
        #         print("Depth", i, j, depth[i][j])
                
        print("depth info", numpy.mean(depth.astype(numpy.uint16)), depth.astype(numpy.uint16)[0][0])
        depthTransform = cv2.warpPerspective(depth.astype(numpy.uint16), numpy.linalg.inv(self.homography), (gray.shape[1], gray.shape[0])) # Might(?) need a different shape
        
        cv2.imshow("DT", depthTransform)
        wall = numpy.bitwise_and(depthTransform, depthTransform, self.backgroundMask) # Need to ensure types work
        cv2.imshow("Wall", wall)
        
        wall = wall[wall > 0.1]
        wall = wall[wall < numpy.inf]
        wallThresh = float(numpy.mean(wall)) * self.threshScale # Not sure if this is 100% right
        # cv2.imshow("WT", wallThresh)
        _, wallMask = cv2.threshold(depthTransform, wallThresh, 255, cv2.THRESH_BINARY) # Might want to use wall here
        # cv2.imshow("WM", wallMask)
        maskedImage = cv2.bitwise_and(self.project, self.project, mask=wallMask.astype(numpy.uint8))
        # cv2.imshow("MI", maskedImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return maskedImage

    def updateProjection(self, image):
        # Get the dimensions of the original image
        img_height, img_width, _ = image.shape

        # Calculate how many tiles are needed in each dimension
        horizontal_tiles = (1920 // img_width) + 1  # +1 to ensure full coverage
        vertical_tiles = (1080 // img_height) + 1  # +1 to ensure full coverage

        # Create a larger image by repeating the original image
        # Tile the image horizontally and vertically
        tiled_image = numpy.tile(image, (vertical_tiles, horizontal_tiles, 1))
        tiled_image = cv2.resize(tiled_image, (1920, 1080))

        # Crop the image to the exact target size
        tiled_image = tiled_image[:1080, :1920]

        self.project = tiled_image

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
        # Convert the image to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # if len(self.foregroundMask) == 0:
        #     self.foregroundMask = numpy.full_like(img, (255, 255, 255), dtype=numpy.uint8)
        #     self.foregroundMask = cv2.cvtColor(self.foregroundMask, cv2.COLOR_BGR2GRAY)

        # Find the distance between the expected projection and what we see
        # else:
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     proj = cv2.drawContours(gray, [self.dst], -1, 255, thickness=cv2.FILLED)
        #     tmp = numpy.full_like(gray, 255, dtype=numpy.uint8)
        #     mask = cv2.bitwise_not(tmp, proj)
        #     gray[mask==255] = mask[mask==255]
        #     transform = cv2.warpPerspective(gray, numpy.linalg.inv(self.homography), (self.last.shape[1], self.last.shape[0]))
        #     avr = numpy.mean(numpy.abs(cv2.subtract(transform.astype(numpy.int16), self.last.astype(numpy.int16))))
        #     if avr < self.differenceThresh:
        #         return

        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(hsv, (3, 3), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=200, apertureSize=3)

        # Dilate and erode
        k1 = numpy.ones((1, 1), numpy.uint8)
        kernel = numpy.ones((3, 3), numpy.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_ERODE, k1, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=3)
        
        # Draw in known bounds for contour identification: screen and edge of image
        edges = cv2.drawContours(edges, [self.dst], -1, (255, 255, 255), 5)
        edges = cv2.rectangle(edges, (0, 0), (edges.shape[1], edges.shape[0]), (255, 255, 255), 5)

        # cv2.imshow("Edges", edges)

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
            edges = cv2.drawContours(edges, [approx_polygon], -1, (255, 255, 255), 5)

            # # Poly lines approach
            # pts = contour.reshape((-1, 1, 2))
            # edges = cv2.polylines(edges, [pts], True, (255, 255, 255), 10)
        
        # edges = cv2.drawContours(edges, [self.dst], -1, (255, 255, 255), 10

        # cv2.imshow("Contours", edges)

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

        # cv2.imshow("c", contour_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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

        cd = ContourDetector(numpy.int32([[[0, 0]], [[0, 900]], [[1200, 900]], [[1200, 0]]]), None)
        # cv2.imshow("Img", cd.project)
        # cd.processFrame(image)
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

        # cv2.imshow("", cd.foregroundMask)
        # cv2.imshow("Big Mask", cv2.bitwise_and(cd.foregroundMask, cd.backgroundMask))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    ContourDetector.test()