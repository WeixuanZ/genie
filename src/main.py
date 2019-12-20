import os
import sys
import argparse
import random
import numpy as np
import cv2
import time

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras

from Preprocessor import *


class FilePaths:
	raw = '../data/raw/'
	gray = '../data/gray/'
	gray_train = '../data/gray/train/'
	gray_test = '../data/gray/test/'
	threshed = '../data/threshed/'
	threshed_train = '../data/threshed/train/'
	threshed_test = '../data/threshed/test/'





def main():
	


