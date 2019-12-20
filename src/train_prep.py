import os
import sys
import argparse
import random
import numpy as np
import cv2

from Preprocessor import *

class FilePaths:
	raw = '../data/raw/'
	gray = '../data/gray/'
	gray_train = '../data/gray/train/'
	gray_test = '../data/gray/test/'
	threshed = '../data/threshed/'
	threshed_train = '../data/threshed/train/'
	threshed_test = '../data/threshed/test/'


parser = argparse.ArgumentParser(prog='train_prep',usage='%(prog)s [options] path', description='Prepare the training data by extracing the roi.')

parser.add_argument('Path', metavar='path', type=str, nargs='+', help='the path to raw images')
parser.add_argument('-d', '--divide', type=float, default=0.95, help='divide the images into a training set and a test set')


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


for path in input_path:
	roi_gray, roi_thresh = extract_roi(read_img(path))
	cv2.imwrite(FilePaths.gray+os.path.basename(path), roi_gray)
	cv2.imwrite(FilePaths.threshed+os.path.basename(path), roi_thresh)
	printProgressBar(input_path.index(path), len(input_path))




