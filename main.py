import cv2 
import numpy as np 
from screeninfo import get_monitors

monitors = get_monitors()
projector = monitors[1]
refImages = ["images/pattern1.png", "images/pattern2.png", "images/pattern3.png"]

def calibrate(imgIndex: int):
	try:
		# Open image
		refImg = cv2.imread(refImages[imgIndex], cv2.IMREAD_COLOR)
		
		# Create a window and move it to the projector screen
		cv2.namedWindow("ProjectorWindow", cv2.WINDOW_FULLSCREEN)
		cv2.moveWindow("ProjectorWindow", projector.x, 0)

		# Now set the window to full-screen mode
		cv2.setWindowProperty("ProjectorWindow", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

		# Display the image on the projector screen
		cv2.imshow("ProjectorWindow", refImg)

		# Keep the window open until a key is pressed
		cv2.waitKey(0)

		# initializing web cam  
		cam = cv2.VideoCapture(0) 

		# creating the SIFT algorithm 
		sift = cv2.SIFT_create() 

		# find the keypoints and descriptors with SIFT 
		kp_image, desc_image = sift.detectAndCompute(refImg, None) 

		# initializing the dictionary 
		index_params = dict(algorithm = 0, trees = 5) 
		search_params = dict() 

		# by using Flann Matcher 
		flann = cv2.FlannBasedMatcher(index_params, search_params) 

		# reading the frame 
		_, frame = cam.read() 

		# converting the frame into grayscale 
		grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

		# find the keypoints and descriptors with SIFT 
		kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None) 

		# finding nearest match with KNN algorithm 
		matches= flann.knnMatch(desc_image, desc_grayframe, k=2) 

		# initialize list to keep track of only good points 
		good_points=[] 

		for m, n in matches: 
			#append the points according 
			#to distance of descriptors 
			if(m.distance < 0.6*n.distance): 
				good_points.append(m) 

		# maintaining list of index of descriptors 
		# in query descriptors 
		query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2) 

		# maintaining list of index of descriptors 
		# in train descriptors 
		train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2) 

		# finding perspective transformation 
		# between two planes 
		matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0) 

		# ravel function returns 
		# contiguous flattened array 
		matches_mask = mask.ravel().tolist() 

		# initializing height and width of the image 
		h, w = refImg.shape[:2]

		# saving all points in pts 
		pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2) 

		# applying perspective algorithm 
		dst = cv2.perspectiveTransform(pts, matrix) 

		# using drawing function for the frame 
		homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3) 

		# showing the final output 
		# with homography 
		cv2.imshow("Homography", homography) 

		# cv2.imshow('Camera', frame)

		cv2.waitKey(0)
		cv2.destroyAllWindows()

	except Exception as e:
		print(e)
		cv2.destroyAllWindows()
		calibrate((imgIndex + 1) % len(refImages)) # Try the next reference image in a circular array

	# finally:
	# 	cv2.destroyAllWindows()

def main():
    calibrate(0)
	
if __name__ == "__main__":
    main()