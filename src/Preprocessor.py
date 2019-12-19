import cv2
import numpy as np


def img_resize(img, img_size):
	return cv2.resize(img, img_size, interpolation = cv2.INTER_AREA)

def read_img(img_path):
	return cv2.imread(img_path)

def create_emptyimg(img_size):
	return np.zeros([imgSize[1], imgSize[0]])


def preprocess(image, img_size = (152,34), verbose = False):
	'''Function extracting the ROI and preprocessing it.

	If no ROI is detected, empty image of the specified size is returned. Note that the annotated input image is never resized.

	Args:
		param1 (str): The path of input image.
        [param2 (2-tuple): The size of output ROI.]
        [param3 (bool, [False]), specify whether the input image is returned with annotations.]

    Returns:
        bool = False (default): roi_gray, roi_thresh
        bool = True: img, roi_gray, roi_thresh
	'''
	 
	(ih,iw,_) = image.shape
	img = image.copy()

	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# reduce noise
	blur = cv2.GaussianBlur(gray, (7, 7), 0)
	# edge detection using Canny
	canny = cv2.Canny(blur, 50, 150)

	contours, hierarch = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	detected = False
	contour = []

	# loop over the contours
	for cnt in contours:
	    ## Get the stright bounding rect
		x,y,w,h = cv2.boundingRect(cnt)
		(cx,cy) = (int(x+w/2), int(y+h/2))

		if (w > 100 or h > 30 or w*h > 3000) and (w < 300 and h < 100) and (iw/2-0.1*iw < cx < iw/2+0.1*iw and ih/2-0.1*ih < cy < ih/2+0.1*ih):
			contour.append(cnt)
			detected = True

			break

	if detected is True:

		img_roi = image[y:y+h, x:x+w]
		roi_gray = cv2.bitwise_not(cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY))
		roi_blur = cv2.medianBlur(roi_gray,3)

		roi_thresh = cv2.adaptiveThreshold(roi_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)

		kernel = np.ones((2,2),np.uint8)
		roi_thresh = cv2.dilate(roi_thresh,kernel,iterations = 2)
		roi_thresh = cv2.erode(roi_thresh,kernel,iterations = 1)

		if verbose is False:
			return img_resize(roi_gray,img_size), img_resize(roi_thresh,img_size)
		else:
			# Reference
			cv2.drawMarker(img, (int(iw/2),int(ih/2)), (0,255,0),markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_AA)
			cv2.rectangle(img, (int(iw/2-0.1*iw),int(ih/2-0.1*ih)), (int(iw/2+0.1*iw),int(ih/2+0.1*ih)), (0,255,0), 2)
			# draw contour
			cv2.drawContours(img, contour, -1, (255, 0, 0), 2)

			## Draw rect
			cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2, 16)
			cv2.drawMarker(img, (cx,cy), (0,0,255),markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_AA)

			return img, img_resize(roi_gray,img_size), img_resize(roi_thresh,img_size)

	
	else:

		empty_img = create_emptyimg(img_size)

		if verbose is False:
			return img_resize(empty_img,img_size), img_resize(empty_img,img_size)
		else:
			cv2.putText(img, 'No meter reading detected', (int(iw/2-200),int(ih/2-100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
			return img, img_resize(empty_img,img_size), img_resize(empty_img,img_size)




if __name__ == '__main__':

	img, roi_gray, roi_thresh = preprocess(read_img('test.png'), verbose=True)

	print(roi_gray.shape)

	cv2.imshow('Image',img)
	cv2.imshow('ROI',roi_gray)
	cv2.imshow('Threshed',roi_thresh)

	cv2.waitKey(0)







