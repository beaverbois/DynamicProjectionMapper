import cv2
import numpy
from consts import Consts
from screeninfo import get_monitors

class ContourDetector():
    blurXMod = 0.005
    blurYMod = 0.005
    
    dilateXMod = 0.02
    dilateYMod = 0.01

    def __init__(self, img, dst):
        # Get values from dst

        print(dst)
        # dst = numpy.array(dst)

        dstMinX = dst[0][0][0]
        dstMinY = dst[0][0][1]
        dstMaxX = dst[0][0][0]
        dstMaxY = dst[0][0][1]


        for i in range(len(dst)):
            dstMinX = min(dstMinX, dst[i][0][0])
            dstMaxX = max(dstMaxX, dst[i][0][0])
            dstMinY = min(dstMinY, dst[i][0][1])
            dstMaxY = max(dstMaxY, dst[i][0][1])

        xDist = dstMaxX - dstMinX
        yDist = dstMaxY - dstMinY

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=200, apertureSize=3)

        # Dilate and erode
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, (9, 9), iterations=19)
        
        edges = cv2.drawContours(edges, [dst], -1, (255, 255, 255), 2)

        # Show the original and edge-detected images
        # cv2.imshow('Original Image', img)
        cv2.imshow('Canny Edge Detection', edges)

        # Find contours from the edges image
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original image to draw the contours on
        contour_image = numpy.zeros_like(edges, dtype=numpy.uint8)
        # maxContour_image = image.copy()
        # flooded_image = image.copy()

        for contour in contours:
            # Get the bounding box of each contour
            x, y, w, h = cv2.boundingRect(contour)

            if w * h > int((xDist * yDist) / 1.5):
                continue

            # cv2.rectangle(edges, (x, y), (x + w, y + h), (255, 255, 255), 2)
            perimeter = cv2.arcLength(contour, True)
        
            # Set epsilon to 2% of the perimeter (you can adjust this for more/less simplification)
            epsilon = 0.05 * perimeter  # This controls the approximation accuracy
        
            # Approximate the contour
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

            edges = cv2.drawContours(edges, [approx_polygon], -1, (255, 255, 255), 2)
        
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, (9, 9), iterations=19)
        
        edges = cv2.drawContours(edges, [dst], -1, (255, 255, 255), 2)

        # cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
        cv2.imshow("Round 1", edges)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through each contour
        maxSize = 0
        maxContour = None
        numContours = 0
        for contour in contours:
            # Get the bounding box of each contour
            x, y, w, h = cv2.boundingRect(contour)

            if w*h > maxSize and w*h < xDist * yDist:
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
        cv2.drawContours(img, [maxContour], -1, (255, 255, 255), cv2.FILLED)
        cv2.imshow("Contour", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.drawContours(contour_image, [maxContour], -1, (255, 255, 255), cv2.FILLED)

        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_HSV2BGR)
        self.mask = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)

    def scaleImage(self, homography):
        # Koala
        projection = cv2.imread("images/pattern3.png")
        
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

        mask_transform = cv2.warpPerspective(self.mask, numpy.linalg.inv(homography), (projection.shape[0], projection.shape[1])) # Set this variable

        # Extract the region inside the contour from the source image using the mask
        contour_region = cv2.bitwise_and(projection, projection, mask=mask_transform)

        # # Use the inverse mask to keep the background of the target image
        # background = cv2.bitwise_and(contour_region, contour_region, mask=inverse_mask)

        # # Overlay the contour region onto the target image
        # result = cv2.add(background, contour_region)

        cv2.imshow("Output", contour_region)

        cv2.imshow("Mask", mask_transform)

        # cv2.imshow("Image with Contours", contour_image)
        # # cv2.imshow("Image Flooded", flooded_image)

        # Wait for a key press and close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

        ContourDetector(image, [(0, 0), (0, 900), (1600, 900), (1600, 0)])