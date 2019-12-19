import time
import cv2
import numpy as np

image = cv2.imread('test.png')
(ih,iw,_) = image.shape
print(iw,ih)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# reduce noise
blur = cv2.GaussianBlur(gray, (13, 13), 0)
# edge detection using Canny
canny = cv2.Canny(blur, 50, 150)


cv2.imshow('Edged',canny)


contours, hierarch = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnts = []
bboxes = []
# rboxes = []

# loop over the contours
for cnt in contours:
    ## Get the stright bounding rect
    bbox = cv2.boundingRect(cnt)
    x,y,w,h = bbox
    (cx,cy) = (int(x+w/2), int(y+h/2))

    if (w > 100 or h > 30 or w*h > 3000) and (w < 300 and h < 100) and (iw/2-0.1*iw < cx < iw/2+0.1*iw and ih/2-0.1*ih < cy < ih/2+0.1*ih):

	    ## Draw rect
	    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2, 16)
	    cv2.drawMarker(image, (cx,cy), (0,0,255),markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_AA)

	    # ## Get the rotated rect
	    # rbox = cv2.minAreaRect(cnt)
	    # (cx,cy), (w,h), rot_angle = rbox
	    # print("rot_angle:", rot_angle)  

	    ## backup 
	    bboxes.append(bbox)
	    # rboxes.append(rbox)
	    cnts.append(cnt)

	    break


cv2.drawContours(image, cnts, -1, (255, 0, 0), 2)

cv2.drawMarker(image, (int(iw/2),int(ih/2)), (0,255,0),markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3, line_type=cv2.LINE_AA)
cv2.rectangle(image, (int(iw/2-0.1*iw),int(ih/2-0.1*ih)), (int(iw/2+0.1*iw),int(ih/2+0.1*ih)), (0,255,0), 2)




cv2.imshow('Image',image)
cv2.waitKey(0)