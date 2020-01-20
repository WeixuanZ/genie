# Gas meter monitoring using OpenCV and CNN

1. ROI extraction with thresholding and edge detection

	![ROI extraction](/doc/roi.png)
	
1. Digit extraction using MSER 

	![Digit extraction](/doc/digits.png)

1. VGG-like CNN trained on SVHN dataset

	![Model](/doc/model.png)

1. Post-processing
	Using a median filter and convolution with FFT

	![Model](/doc/results.png)

## TODO

- [ ] Check digit extraction order, maybe using a mask instead of Euclidean distance
