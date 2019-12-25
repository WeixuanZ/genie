import cv2
import numpy as np


def img_resize(img, img_size):
    """Function creates empty np.arrays of the specified dimensions to act as blank image
    Args:
        param1 (np.array): image to resize
        param2 (int, int): (x, y) dimensions of output image
    Returns:
        (np.array): image of requested dimensions
    """
    return cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)


def read_img(img_path):
    return cv2.imread(img_path)


def create_emptyimg(img_size):
    """Function creates empty np.arrays of the specified dimensions to act as blank image
    Args:
        param1 (int, int): (x, y) dimensions of blank image
    Returns:
        np.array[int][int] of zeros
    """
    return np.zeros([img_size[1], img_size[0]], np.uint8)


def extract_roi(image, img_size=(250, 30), verbose=False):
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

    (ih, iw, _) = image.shape
    img = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_thresh = np.uint8(np.where(gray > 140, 255, 0))
    kernel = np.ones((5, 5), np.uint8)
    gray_thresh = cv2.erode(gray_thresh, kernel, iterations=2)
    gray_thresh = cv2.dilate(gray_thresh, kernel, iterations=2)
    # # reduce noise
    # blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # blur = gray
    # edge detection using Canny
    canny = cv2.Canny(gray_thresh, 50, 150)

    contours, hierarch = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    contour = []

    # loop over the contours
    for cnt in contours:
        ## Get the stright bounding rect
        x, y, w, h = cv2.boundingRect(cnt)
        (cx, cy) = (int(x + w / 2), int(y + h / 2))

        if (w > 100 and h > 25) and (w < 300 and h < 100) and (
                iw / 2 - 0.05 * iw < cx < iw / 2 + 0.05 * iw and ih / 2 - 0.05 * ih < cy < ih / 2 + 0.05 * ih):
            contour.append(cnt)
            detected = True

            break

    if detected is True:

        img_roi = image[y:y + h, x:x + w]
        roi_gray = cv2.bitwise_not(cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY))
        roi_blur = cv2.medianBlur(roi_gray, 3)

        roi_thresh = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        roi_thresh = cv2.medianBlur(roi_thresh, 3)

        kernel = np.ones((2, 2), np.uint8)
        roi_thresh = cv2.dilate(roi_thresh, kernel, iterations=2)
        roi_thresh = cv2.erode(roi_thresh, kernel, iterations=1)
        # roi_thresh = cv2.medianBlur(roi_thresh, 3)

        if verbose is False:
            return img_resize(roi_gray, img_size), img_resize(roi_thresh, img_size)
        else:
            # Reference
            cv2.drawMarker(img, (int(iw / 2), int(ih / 2)), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20,
                           thickness=3, line_type=cv2.LINE_AA)
            cv2.rectangle(img, (int(iw / 2 - 0.05 * iw), int(ih / 2 - 0.05 * ih)),
                          (int(iw / 2 + 0.05 * iw), int(ih / 2 + 0.05 * ih)), (0, 255, 0), 2)
            # draw contour
            cv2.drawContours(img, contour, -1, (255, 0, 0), 2)

            ## Draw rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2, 16)
            cv2.drawMarker(img, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3,
                           line_type=cv2.LINE_AA)

            return img, img_resize(roi_gray, img_size), img_resize(roi_thresh, img_size)


    else:

        empty_img = create_emptyimg(img_size)

        if verbose is False:
            return img_resize(empty_img, img_size), img_resize(empty_img, img_size)
        else:
            cv2.putText(img, 'No meter reading detected', (int(iw / 2 - 200), int(ih / 2 - 100)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return img, img_resize(empty_img, img_size), img_resize(empty_img, img_size)


def remove_border(image):
    image[:3, :] = 255
    image[-3:, :] = 255
    image[:, :3] = 255
    image[:, -3:] = 255
    return image


def Euclidean(vec1, vec2):
    """
    Euclidean_Distance
    :param vec1:
    :param vec2:
    :return:
    """
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return np.sqrt(((npvec1 - npvec2) ** 2).sum())


def extract_digit(image, inc_last=False):
    mser = cv2.MSER_create(_min_area=50, _max_area=100)
    regions, boxes = mser.detectRegions(image)
    fewer_boxes = []
    distinct_boxes = []
    digits = []

    # removing duplicates
    for box in boxes.tolist():
        if box not in fewer_boxes:
            fewer_boxes.append(box)

    # print(len(fewer_boxes))

    # removing similar
    for j in fewer_boxes:
        if distinct_boxes:
            for k in range(len(distinct_boxes)):
                if Euclidean(j, distinct_boxes[k]) > 10 and k == len(distinct_boxes) - 1:
                    distinct_boxes.append(j)
        else:
            distinct_boxes.append(j)

    # print(len(distinct_boxes))

    # extracting digits and store to a list
    for box in distinct_boxes[:8]:
        x, y, w, h = box
        digit = image[y - 2:y + h + 2, x - 2:x + w + 2]
        digit = img_resize(cv2.copyMakeBorder(digit, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255), (32, 32))
        # digit = cv2.adaptiveThreshold(digit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        digits.append(digit)

    # for i in range(len(digits)):
    #     cv2.imshow(str(i),digits[i])

    return digits[:7] if inc_last else digits


if __name__ == '__main__':
    img, roi_gray, roi_thresh = extract_roi(read_img('test.png'), verbose=True)

    # print(roi_gray.shape)

    # cv2.imshow('Image', img)
    # cv2.imshow('ROI', roi_gray)
    # cv2.imshow('Threshed', remove_border(roi_thresh))

    digits = extract_digit(remove_border(roi_thresh))

    for i in range(len(digits)):
        # digits[i] = cv2.adaptiveThreshold(digits[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        # kernel = np.ones((5, 5))
        # digits[i] = cv2.erode(digits[i], kernel, iterations=1)
        # digits[i] = cv2.blur(digits[i], (5, 5))
        # text = pytesseract.image_to_string(digits[i])
        # print(text)
        cv2.imwrite('./' + str(i) + '.png', digits[i])

    cv2.waitKey(0)
