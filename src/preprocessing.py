import time
import cv2
import numpy as np

image = cv2.imread('test.png')
(ih,iw,_) = image.shape
img = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# reduce noise
blur = cv2.GaussianBlur(gray, (7, 7), 0)
# edge detection using Canny
canny = cv2.Canny(blur, 50, 150)

# cv2.imshow('Edged',canny)


contours, hierarch = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# rboxes = []
detected = False
contour = []

# loop over the contours
for cnt in contours:
    ## Get the stright bounding rect
	x,y,w,h = cv2.boundingRect(cnt)
	(cx,cy) = (int(x+w/2), int(y+h/2))

	if (w > 100 or h > 30 or w*h > 3000) and (w < 300 and h < 100) and (iw/2-0.1*iw < cx < iw/2+0.1*iw and ih/2-0.1*ih < cy < ih/2+0.1*ih):

	    # ## Get the rotated rect
	    # rbox = cv2.minAreaRect(cnt)
	    # (cx,cy), (w,h), rot_angle = rbox
	    # print("rot_angle:", rot_angle)  
	    # rboxes.append(rbox)

		contour.append(cnt)
		detected = True

		break


# Reference
cv2.drawMarker(img, (int(iw/2),int(ih/2)), (0,255,0),markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_AA)
cv2.rectangle(img, (int(iw/2-0.1*iw),int(ih/2-0.1*ih)), (int(iw/2+0.1*iw),int(ih/2+0.1*ih)), (0,255,0), 2)


if detected is True:

	# draw contour
	cv2.drawContours(img, contour, -1, (255, 0, 0), 2)

	## Draw rect
	cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2, 16)
	cv2.drawMarker(img, (cx,cy), (0,0,255),markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_AA)


	cv2.imshow('Image',img)

	img_roi = image[y:y+h, x:x+w]
	cv2.imshow('ROI',img_roi)

	roi_gray = cv2.bitwise_not(cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY))
	# roi_blur = cv2.GaussianBlur(roi_gray,(5,5),0)
	roi_blur = cv2.medianBlur(roi_gray,3)
	# roi_blur = cv2.bilateralFilter(roi_gray,5,75,75)
	# cv2.imshow('Gray',roi_gray)
	# cv2.imshow('Blurred',roi_blur)

	roi_thresh = cv2.adaptiveThreshold(roi_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
	# roi_thresh2 = cv2.bitwise_not(cv2.adaptiveThreshold(roi_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2))
	# roi_thresh2 = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	# cv2.imshow('Threshed_raw',roi_thresh)
	# cv2.imshow('Threshed_raw2',roi_thresh2)

	kernel = np.ones((2,2),np.uint8)
	# roi_thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_OPEN, kernel)
	roi_thresh = cv2.dilate(roi_thresh,kernel,iterations = 2)
	roi_thresh = cv2.erode(roi_thresh,kernel,iterations = 1)
	cv2.imshow('Threshed',roi_thresh)
else:
	cv2.putText(img, 'No meter reading detected', (int(iw/2-200),int(ih/2-100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow('Image',img)


cv2.waitKey(0)