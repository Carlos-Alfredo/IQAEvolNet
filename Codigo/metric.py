import numpy as np
import cv2
import math
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr



def MSE(image_original,image_improved):
	
	if len(image_original.shape) == 2:
		image_original = image_original
	else:
		image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

	if len(image_improved.shape) == 2:
		image_improved = image_improved
	else:
		image_improved = cv2.cvtColor(image_improved, cv2.COLOR_BGR2GRAY)

	error = np.subtract(image_original,image_improved)
	sqrtError = np.square(error)
	meanSqrtError = np.mean(sqrtError)
	return meanSqrtError

def EME(img,rowSample,columnSample):
	
	#grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	if len(img.shape) == 2:
		grayImg = img
	else:
		grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#grayImg = cv2.resize(grayImg, (grayImg.shape[0]+rowSample-grayImg.shape[0]%rowSample,grayImg.shape[1]+rowSample-grayImg.shape[1]%rowSample), interpolation=cv2.INTER_AREA)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/rowSample)
	nColumns = int(columnSize/columnSample)
	incompleteRow = math.ceil(rowSize/rowSample) - nRows
	incompleteColumn = math.ceil(columnSize/columnSample) - nColumns

	somatory = 0
	nBlocks = nRows*nColumns
	for i in range(0,nRows):
		for j in range(0,nColumns):
			imax=grayImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample].max()
			imin=grayImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample].min()
			if imin==0:
				imin=1
			if imax==0:
				imax=1
			somatory = somatory + 20*math.log(imax/imin)

	if incompleteColumn==1:
		for i in range(0,nRows):
			imax=grayImg[i*rowSample:(i+1)*rowSample,nColumns*columnSample:columnSize].max()
			imin=grayImg[i*rowSample:(i+1)*rowSample,nColumns*columnSample:columnSize].min()
			if imin==0:
				imin=1
			if imax==0:
				imax=1
			somatory = somatory + 20*math.log(imax/imin)
			nBlocks = nBlocks + 1
	if incompleteRow==1:
		for j in range(0,nColumns):
			imax=grayImg[nRows*rowSample:rowSize,j*columnSample:(j+1)*columnSample].max()
			imin=grayImg[nRows*rowSample:rowSize,j*columnSample:(j+1)*columnSample].min()
			if imin==0:
				imin=1
			if imax==0:
				imax=1
			somatory = somatory + 20*math.log(imax/imin)
			nBlocks = nBlocks + 1
	if incompleteRow==1 and incompleteColumn==1:
		imax=grayImg[nRows*rowSample:rowSize,nColumns*columnSample:columnSize].max()
		imin=grayImg[nRows*rowSample:rowSize,nColumns*columnSample:columnSize].max()
		if imin==0:
			imin=1
		if imax==0:
			imax=1
		somatory = somatory + 20*math.log(imax/imin)
		nBlocks = nBlocks + 1
	return somatory/nBlocks

def EMEE(img,rowSample,columnSample):
	
	#grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	if len(img.shape) == 2:
		grayImg = img
	else:
		grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#grayImg = cv2.resize(grayImg, (grayImg.shape[0]+rowSample-grayImg.shape[0]%rowSample,grayImg.shape[1]+rowSample-grayImg.shape[1]%rowSample), interpolation=cv2.INTER_AREA)
	rowSize, columnSize = grayImg.shape
	nRows = int(rowSize/rowSample)
	nColumns = int(columnSize/columnSample)
	incompleteRow = math.ceil(rowSize/rowSample) - nRows
	incompleteColumn = math.ceil(columnSize/columnSample) - nColumns

	somatory = 0
	nBlocks = nRows*nColumns
	for i in range(0,nRows):
		for j in range(0,nColumns):
			imax=grayImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample].max()
			imin=grayImg[i*rowSample:(i+1)*rowSample,j*columnSample:(j+1)*rowSample].min()
			if imin==0:
				imin=1
			if imax==0:
				imax=1
			somatory = somatory + imax/imin*math.log(imax/imin)

	if incompleteColumn==1:
		for i in range(0,nRows):
			imax=grayImg[i*rowSample:(i+1)*rowSample,nColumns*columnSample:columnSize].max()
			imin=grayImg[i*rowSample:(i+1)*rowSample,nColumns*columnSample:columnSize].min()
			if imin==0:
				imin=1
			if imax==0:
				imax=1
			somatory = somatory + imax/imin*math.log(imax/imin)
			nBlocks = nBlocks + 1
	if incompleteRow==1:
		for j in range(0,nColumns):
			imax=grayImg[nRows*rowSample:rowSize,j*columnSample:(j+1)*columnSample].max()
			imin=grayImg[nRows*rowSample:rowSize,j*columnSample:(j+1)*columnSample].min()
			if imin==0:
				imin=1
			if imax==0:
				imax=1
			somatory = somatory + imax/imin*math.log(imax/imin)
			nBlocks = nBlocks + 1
	if incompleteRow==1 and incompleteColumn==1:
		imax=grayImg[nRows*rowSample:rowSize,nColumns*columnSample:columnSize].max()
		imin=grayImg[nRows*rowSample:rowSize,nColumns*columnSample:columnSize].max()
		if imin==0:
			imin=1
		if imax==0:
			imax=1
		somatory = somatory + imax/imin*math.log(imax/imin)
		nBlocks = nBlocks + 1
	return somatory/nBlocks

def RMSE(image_original,image_improved):

	if len(image_original.shape) == 2:
		image_original = image_original
	else:
		image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

	if len(image_improved.shape) == 2:
		image_improved = image_improved
	else:
		image_improved = cv2.cvtColor(image_improved, cv2.COLOR_BGR2GRAY)

	error = np.subtract(image_original,image_improved)
	sqrtError = np.square(error)
	meanSqrtError = np.mean(sqrtError)
	return math.sqrt(meanSqrtError)

def IEM_filter(image):
	image = image.astype(np.float32)

	(iH, iW) = image.shape[:2]
	output = np.zeros((iH-2, iW-2), dtype="float32")

	for y in np.arange(0, iH-2):
		for x in np.arange(0, iW-2):

			roi = image[y:y + 3, x:x + 3]

			k = roi
			k = (np.abs(k[1][1] - k[0][0]) + np.abs(k[1][1] - k[0][1]) + np.abs(k[1][1] - k[0][2]) +
				np.abs(k[1][1] - k[1][0]) + np.abs(k[1][1] - k[1][2]) +
				np.abs(k[1][1] - k[2][0]) + np.abs(k[1][1] - k[2][1]) + np.abs(k[1][1] - k[2][2]))  

			output[y, x] = k
	return np.sum(output)

def IEM(image_original, image_improved):
	'''
	Image Enhancement Metric(IEM) approximates the contrast and
	sharpness of an image by dividing an image into non-overlapping
	blocks. 

	imageA is the raw image (before histogram equalization per example), and imageB 
	is the image after preprocessing
	'''
	#image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

	if len(image_original.shape) == 2:
		image_original = image_original
	else:
		image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

	if len(image_improved.shape) == 2:
		image_improved = image_improved
	else:
		image_improved = cv2.cvtColor(image_improved, cv2.COLOR_BGR2GRAY)

	valA = IEM_filter(image_original)
	valB = IEM_filter(image_improved)

	return valB/valA

def AG(image_improved):

	#grayImg_improved = cv2.cvtColor(image_improved, cv2.COLOR_BGR2GRAY)

	if len(image_improved.shape) == 2:
		image_improved = image_improved
	else:
		image_improved = cv2.cvtColor(image_improved, cv2.COLOR_BGR2GRAY)

	sobel_x = cv2.Sobel(image_improved, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
	sobel_y = cv2.Sobel(image_improved, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
	sobel_abs = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)

	averageGradient = np.mean(sobel_abs)/math.sqrt(2)

	return averageGradient

def CNR(image_improved):
	image_improved = cv2.cvtColor(image_improved, cv2.COLOR_BGR2GRAY)
	# Calculate the contrast
	mean, std = cv2.meanStdDev(image_improved)
	contrast = std/mean

	# Create a noisy version of the image
	noise = np.random.normal(0, 1, image_improved.shape)
	noisy = (image_improved + noise).astype(int)

	# Calculate the PSNR
	mse = np.mean((image_improved - noisy) ** 2)
	psnr = 10 * np.log10((255 ** 2) / mse)

	cnr = contrast / psnr

	return cnr

def CNR_IMPROVEMENT(image_original,image_improved):
	return CNR(image_improved)/CNR(image_original)

def SSIM(image_original,image_improved):
	if(len(image_original.shape)==3 and image_original.shape[2]!=1):
		image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
	if(len(image_improved.shape)==3 and image_improved.shape[2]!=1):
		image_improved = cv2.cvtColor(image_improved, cv2.COLOR_BGR2GRAY)
	return ssim(image_original, image_improved)

def PSNR(image_original, image_improved):
	if(len(image_original.shape)==3 and image_original.shape[2]!=1):
		image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
	if(len(image_improved.shape)==3 and image_improved.shape[2]!=1):
		image_improved = cv2.cvtColor(image_improved, cv2.COLOR_BGR2GRAY)
	return psnr(image_original,image_improved)
