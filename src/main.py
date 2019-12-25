import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import argparse
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt

from Preprocessor import *
from Postprocessor import *

parser = argparse.ArgumentParser(prog='main', usage='%(prog)s [options] path',
                                 description='Recognise gas meter reading')
parser.add_argument('--clear', action='store_true', help='clear the output directory')
parser.add_argument('-s', '--save', action='store_true', help='save the recognised data as csv')
parser.add_argument('Path', metavar='path', type=str, nargs='+', help='the path to raw images')

args = parser.parse_args()
input_path = args.Path
s
model = keras.models.load_model('../model/detector_model_2.hdf5')

result = []


# Print iterations progress from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
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
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if args.clear:
    print("Removing all the images")
    os.system('(cd {} && rm ./*.png)'.format('../data/result/'))

else:
    for path in input_path:
        roi_gray, roi_thresh = extract_roi(read_img(path))
        try:
            digits = extract_digit(remove_border(roi_thresh))
            texts = []

            for i in range(len(digits)):
                digit = digits[i]
                kernel = np.ones((2, 2), np.uint8)
                digit = cv2.dilate(digit, kernel, iterations=1)
                digit = np.dstack([digit, digit, digit])
                digit = digit.reshape((1, 32, 32, 3))
                prediction = model.predict(digit)
                digit_predict = np.argmax(prediction)
                if i == 0:
                    digit_predict = 0
                if digit_predict == 10:
                    digit_predict = 0
                texts.append(str(digit_predict))
        except:
            texts = ['0']

        print(''.join(texts))
        result.append(int(''.join(texts)))
        cv2.putText(roi_gray, str(''.join(texts)), (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite('../data/result/' + os.path.basename(path), roi_gray)
        # cv2.imwrite('../data/result/threshed/' + os.path.basename(path), roi_thresh)

        printProgressBar(input_path.index(path), len(input_path), prefix='Progress:', suffix='Complete')

    if args.save:
        np.savetxt('../data/data.csv', result, delimiter=',')

    plt.plot(np.arange(1, len(input_path) + 1, 1), smooth(result))
    plt.ylim([7.5e6, 7.6e6])
    plt.ylabel('Reading')
    plt.xlabel('Image Index')
    plt.show()
