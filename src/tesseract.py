import os
import argparse
import numpy as np
import cv2
import pytesseract

from Preprocessor import *

# roi = (255, 300, 255, 28)

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




if args.clear:
	print("Removing all the images")
	os.system('(cd {} && rm ./*.png)'.format('../data/result/'))

else:
	for path in input_path:
		roi_gray, roi_thresh = extract_roi(read_img(path))
		digits = extract_digit(remove_border(roi_thresh))
		texts = []
		# print(digits)

		for i in digits:
			i = img_resize(i,(100,100))
			i = cv2.adaptiveThreshold(i, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
			text = pytesseract.image_to_string(i, config='outputbase digits')
			text = filter(lambda char: char not in " -?.!/;:", text)
			text = "".join(list(text))
			print(text)
			texts.append(text)

		cv2.putText(roi_gray, str(''.join(texts)), (2,2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
		
		if args.verbose:
			cv2.imshow("Threshed",roi_thresh)
			cv2.imshow("Reading",roi_gray)
			print(pytesseract.image_to_string(roi_thresh, config='outputbase digits'))
			cv2.waitKey(0)
		else:
			cv2.imwrite('../data/result/' + os.path.basename(path), roi_gray)
			cv2.imwrite('../data/result/threshed/' + os.path.basename(path), roi_thresh)

		printProgressBar(input_path.index(path), len(input_path), prefix = 'Progress:', suffix = 'Complete')


