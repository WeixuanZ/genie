import os
import argparse
import numpy as np
import cv2

from Preprocessor import *


parser = argparse.ArgumentParser(prog='tesseract',usage='%(prog)s [options] path', description='Prepare the training data by extracing the roi.')
parser.add_argument('--clear', action='store_true', help='clear the output directory')
parser.add_argument('-v', '--verbose', action='store_true', help='clear the output directory')

parser.add_argument('Path', metavar='path', type=str, nargs='+', help='the path to raw images')

args = parser.parse_args()
input_path = args.Path



# Print iterations progress from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Modified for Felix's code
def remove_border(img):
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
	for i, row in enumerate(img):
		if np.count_nonzero(row) < 250:
		    img[i,:] = np.ones(500, np.uint8) * 255
	for i, col in enumerate(img[:]):
		if np.count_nonzero(col) < 30:
		    img[:, i] = np.ones(100, np.uint8) * 255

	return img





if args.clear:
	print("Removing all the images")
	os.system('(cd {} && rm ./*.png)'.format(FilePaths.gray))
	os.system('(cd {} && rm ./*.png)'.format(FilePaths.gray_test))
	os.system('(cd {} && rm ./*.png)'.format(FilePaths.gray_train))
	os.system('(cd {} && rm ./*.png)'.format(FilePaths.threshed))
	os.system('(cd {} && rm ./*.png)'.format(FilePaths.threshed_test))
	os.system('(cd {} && rm ./*.png)'.format(FilePaths.threshed_train))

else:
	for path in input_path:
		roi_gray, roi_thresh = extract_roi(read_img(path), img_size=(500,100))

		# roi_thresh = remove_border(roi_thresh)

		text = pytesseract.image_to_string(roi_thresh, config='outputbase digits')
		text = filter(lambda char: char not in " -?.!/;:", text)
		text = "".join(list(text))

		cv2.putText(roi_gray, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		
		if args.verbose:
			cv2.imshow("Threshed",roi_thresh)
			cv2.imshow("Reading",roi_gray)
			print(pytesseract.image_to_string(roi_thresh, config='outputbase digits'))
			cv2.waitKey(0)
		else:
			cv2.imwrite('../data/result/' + os.path.basename(path), roi_gray)

		printProgressBar(input_path.index(path), len(input_path), prefix = 'Progress:', suffix = 'Complete')


