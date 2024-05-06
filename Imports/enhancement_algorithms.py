import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
# from huggingface_hub import from_pretrained_keras
# from transformers import AutoModel
from PIL import Image
# import tensorflow as tf
# from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import OrderedDict
import PIL
import pywt

def normalize(min_old, max_old, min_new, max_new, val):
	'''Normalizes values to the interval [min_new, max_new]

	Parameters:
		min_old: min value from old base.
		max_old: max value from old base.
		min_new: min value from new base.
		max_new: max value from new base.
		val: float or array-like value to be normalized.
	'''

	ratio = (val - min_old) / (max_old - min_old)
	normalized = (max_new - min_new) * ratio + min_new
	return normalized.astype(np.uint8)

def histogram(data):
	'''Generates the histogram for the given data.
	Parameters:
		data: data to make the histogram.
	Returns: histogram, bins.
	'''

	pixels, count = np.unique(data, return_counts=True)
	hist = OrderedDict()

	for i in range(len(pixels)):
		hist[pixels[i]] = count[i]

	return np.array(list(hist.values())), np.array(list(hist.keys()))

def UM(image,raio_gauss,fator_mask):
	blur_img = cv2.GaussianBlur(image, (raio_gauss,raio_gauss), 0)
	mask = cv2.subtract(image, blur_img)
	sharp_img = cv2.addWeighted(image, 1, mask, fator_mask, 0)
	return sharp_img.astype(np.uint8)

def HEF(image,d0v):
	img = image
	
	img = normalize(np.min(img), np.max(img), 0, 255, img)

	img_fft = fft2(img)  # img after fourier transformation
	img_sfft = fftshift(img_fft)  # img after shifting component to the center

	#sharp_img = cv2.addWeighted(image, 1, mask, fator_mask, 0)

	m, n = img_sfft.shape
	filter_array = np.zeros((m, n))

	for i in range(m):
		for j in range(n):
			filter_array[i, j] = 1.0 - np.exp(- ((i-m / 2.0) ** 2 + (j-n / 2.0) ** 2) / (2 * (d0v ** 2)))
	k1 = 0.5
	k2 = 0.75
	high_filter = k1 + k2*filter_array

	img_filtered = high_filter * img_sfft
	img_hef = np.real(ifft2(fftshift(img_filtered)))  # HFE filtering done

	# HE part
	# Building the histogram
	hist, bins = histogram(img_hef)
	# Calculating probability for each pixel
	pixel_probability = hist / hist.sum()
	# Calculating the CDF (Cumulative Distribution Function)
	cdf = np.cumsum(pixel_probability)
	cdf_normalized = cdf * 255
	hist_eq = {}
	for i in range(len(cdf)):
		hist_eq[bins[i]] = int(cdf_normalized[i])

	enhanced_image = np.zeros(image.shape)
	for i in range(m):
		for j in range(n):
			enhanced_image[i][j] = hist_eq[img_hef[i][j]]

	return enhanced_image.astype(np.uint8)

def CLAHE(image,clipLimit,raio):
	clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(raio,raio))
	return clahe.apply(image)



def mod_padding_symmetric(image, factor=64):
	"""Padding the image to be divided by factor."""
	height, width = image.shape[0], image.shape[1]
	height_pad, width_pad = ((height + factor) // factor) * factor, ( (width + factor) // factor) * factor
	padh = height_pad - height if height % factor != 0 else 0
	padw = width_pad - width if width % factor != 0 else 0
	image = tf.pad(image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)], mode="REFLECT")
	return image


def make_shape_even(image):
	"""Pad the image to have even shapes."""
	height, width = image.shape[0], image.shape[1]
	padh = 1 if height % 2 != 0 else 0
	padw = 1 if width % 2 != 0 else 0
	image = tf.pad(image, [(0, padh), (0, padw), (0, 0)], mode="REFLECT")
	return image

def process_image(image: Image):
	image = image.resize((256,256))
	input_img = np.asarray(image) / 255.0
	height, width = input_img.shape[0], input_img.shape[1]

	# Padding images to have even shapes
	input_img = make_shape_even(input_img)
	height_even, width_even = input_img.shape[0], input_img.shape[1]

	# padding images to be multiplies of 64
	input_img = mod_padding_symmetric(input_img, factor=64)
	input_img = tf.expand_dims(input_img, axis=0)
	return input_img, height, width, height_even, width_even

def imshow(image, title=None):
	if len(image.shape) > 3:
		image = tf.squeeze(image, axis=0)

	plt.imshow(image,cmap='gray')
	if title:
		plt.title(title)

def infer(image_path: str,new_model):
	image = Image.open(image_path).convert("RGB")
	dim=((np.asarray(image)).shape[1],(np.asarray(image)).shape[0])
	preprocessed_image, height, width, height_even, width_even = process_image(image)

	preds = new_model.predict(preprocessed_image)
	if isinstance(preds, list):
		preds = preds[-1]
		if isinstance(preds, list):
			preds = preds[-1]

	preds = np.array(preds[0], np.float32)

	new_height, new_width = preds.shape[0], preds.shape[1]
	h_start = new_height // 2 - height_even // 2
	h_end = h_start + height
	w_start = new_width // 2 - width_even // 2
	w_end = w_start + width
	preds = preds[h_start:h_end, w_start:w_end, :]
	final_pred_image = cv2.resize(np.array(np.clip(preds, 0.0, 1.0)), dim, interpolation=cv2.INTER_AREA)
	#return np.array(np.clip(preds, 0.0, 1.0))
	return final_pred_image

def maxim_model():
	#model_addr = "google/maxim-s3-deblurring-gopro" #Não funciona
	#model_addr = "google/maxim-s2-enhancement-lol" #Funciona mas é ruim
	model_addr = "google/maxim-s2-dehazing-sots-indoor" #Funciona
	#model_addr = "google/maxim-s3-deblurring-realblur-r" #Não funciona
	#model_addr = "google/maxim-s3-deblurring-reds" #Não funciona
	#model_addr = "google/maxim-s3-deblurring-realblur-j" #Não funciona
	#model_addr = "google/maxim-s3-denoising-sidd" #Não funciona
	#model_addr = "google/maxim-s2-enhancement-fivek" #Funciona
	#model_addr = "google/maxim-s2-dehazing-sots-outdoor" #Não funciona
	#model_addr = "google/maxim-s2-deraining-raindrop" #Não funciona
	#model_addr = "google/maxim-s2-deraining-rain13k" #Não funciona

	new_model = from_pretrained_keras(model_addr)
	#new_model = AutoModel.from_pretrained(model_addr)
	return new_model

def ATACE(xray_img):
	window_size = 3
	
	I = np.asarray(xray_img.astype(np.single))+0.001
	I_nor = I/np.max(I)
	
	local_min = cv2.erode(
		I_nor,cv2.getStructuringElement(
			cv2.MORPH_RECT,(2 * window_size + 1, 2 * window_size + 1)
			)
		)
	local_max = 0.999*cv2.dilate(
		I_nor,cv2.getStructuringElement(
			cv2.MORPH_RECT,(2 * window_size + 1, 2 * window_size + 1)
			)
		)

	removal_factor = np.exp(-1*((local_max*np.var(local_min))/local_min))
	removable_content = local_min*removal_factor

	phi = (np.log(1-removable_content*(np.reciprocal(local_max)-1)))/(np.log(local_max))

	enhanced_image = (I_nor - removable_content)/(np.power(local_max,phi)-removable_content)

	return (enhanced_image*255).astype(np.uint8)

def spacial_frequency(image):
	M,N = image.shape
	image_single = image.astype(np.single)
	frequency_row = (np.sum( ( image_single[:,1:] - image_single[:,:N-1] )** 2)/(M*N)) ** 0.5
	frequency_column = (np.sum( ( image_single[1:,:] - image_single[:M-1,:] )** 2)/(M*N)) ** 0.5
	spacial_freq = (frequency_row**2 + frequency_column**2) ** 0.5
	return spacial_freq

def svd_equalization(image):
	U,S,Vh = np.linalg.svd(img)
	Snorm = S - np.mean(S)
	Snorm = Snorm/np.std(Snorm)
	improvement_factor = Snorm.max()/S.max()
	equalized_image = np.matmul(np.matmul(U,np.diag(improvement_factor*S)),Vh)
	return equalized_image

def TCDHE_SD(image):
	part_point1 = int(np.mean(image) - 0.43*np.std(image))
	part_point2 = int(np.mean(image) + 0.43*np.std(image))
	hist, bins = np.histogram(image,bins=256, range=[0,256])

	hist1 = hist[:part_point1]
	hist2 = hist[part_point1:part_point2]
	hist3 = hist[part_point2:]

	bins1 = bins[:part_point1]
	bins2 = bins[part_point1:part_point2]
	bins3 = bins[part_point2:]

	threshold1 = np.sum(hist1)/np.max(bins1)
	threshold2 = np.sum(hist2)/np.max(bins2)
	threshold3 = np.sum(hist3)/np.max(bins3)

	pdf1 = np.zeros(hist1.shape[0])
	pdf2 = np.zeros(hist2.shape[0])
	pdf3 = np.zeros(hist3.shape[0])
	cdf1 = np.zeros(hist1.shape[0])
	cdf2 = np.zeros(hist2.shape[0])
	cdf3 = np.zeros(hist3.shape[0])
	for i in range(0,hist1.shape[0]):
		pdf1[i] = np.minimum(threshold1,hist1[i])/np.sum(hist1)
		#pdf1[i] = hist1[i]/np.sum(hist1)
		cdf1[i] = np.sum(pdf1[0:i+1])
	for i in range(0,hist2.shape[0]):
		pdf2[i] = np.minimum(threshold2,hist2[i])/np.sum(hist2)
		#pdf2[i] = hist2[i]/np.sum(hist2)
		cdf2[i] = np.sum(pdf2[0:i+1])
	for i in range(0,hist3.shape[0]):
		pdf3[i] = np.minimum(threshold3,hist3[i])/np.sum(hist3)
		#pdf3[i] = hist3[i]/np.sum(hist3)
		cdf3[i] = np.sum(pdf3[0:i+1])

	cdf1 = cdf1/cdf1.max()
	cdf2 = cdf2/cdf2.max()
	cdf3 = cdf3/cdf3.max()

	part_point0 = image.min()
	part_point3 = image.max()

	p0 = 0
	p1 = (part_point1 - part_point0)*255/(part_point3 - part_point0 + 1)
	p2 = (part_point2 - part_point1)*255/(part_point3 - part_point0 + 1) + p1
	p3 = (part_point3 - part_point2 + 1)*255/(part_point3 - part_point0 + 1) + p2

	transfer_function = np.append((p1-1)*cdf1, (p1 + ((p2-1-p1)*cdf2)))
	transfer_function = np.append(transfer_function,( p2 + ((p3-p2)*cdf3)))

	enhanced_image = np.zeros(image.shape)

	for i in range(0,image.shape[0]):
		for j in range(0,image.shape[1]):
			enhanced_image[i,j] = transfer_function[int(image[i,j])].astype(np.uint8)

	return enhanced_image

def TCDHE(I):
	I_line = TCDHE_SD(I)

	coeffs2 = pywt.dwt2(I, 'bior1.3')
	LL, (LH, HL, HH) = coeffs2

	coeffs2 = pywt.dwt2(I_line, 'bior1.3')
	LL_line, (LH_line, HL_line, HH_line) = coeffs2

	U,S,Vh = np.linalg.svd(LL)
	U_line,S_line,Vh_line = np.linalg.svd(LL_line)
	improvement_factor = (U_line.max()+Vh_line.max())/(U.max()+Vh.max())
	beta = 0.5
	Snorm = (beta*improvement_factor*S)+((1-beta)*(1/beta)*S_line)
	LL_norm = np.matmul(np.matmul(U_line,np.diag(Snorm)),Vh_line)

	spacial_freq_LH = spacial_frequency(LH)
	spacial_freq_HL = spacial_frequency(HL)
	spacial_freq_HH = spacial_frequency(HH)
	spacial_freq_LH_line = spacial_frequency(LH_line)
	spacial_freq_HL_line = spacial_frequency(HL_line)
	spacial_freq_HH_line = spacial_frequency(HH_line)

	spacial_freq_LH_norm = spacial_freq_LH/(spacial_freq_LH+spacial_freq_LH_line)
	spacial_freq_HL_norm = spacial_freq_HL/(spacial_freq_HL+spacial_freq_HL_line)
	spacial_freq_HH_norm = spacial_freq_HH/(spacial_freq_HH+spacial_freq_HH_line)
	spacial_freq_LH_line_norm = spacial_freq_LH_line/(spacial_freq_LH+spacial_freq_LH_line)
	spacial_freq_HL_line_norm = spacial_freq_HL_line/(spacial_freq_HL+spacial_freq_HL_line)
	spacial_freq_HH_line_norm = spacial_freq_HH_line/(spacial_freq_HH+spacial_freq_HH_line)

	LH_norm = spacial_freq_LH_norm*LH + spacial_freq_LH_line_norm*LH_line
	HL_norm = spacial_freq_HL_norm*HL + spacial_freq_HL_line_norm*HL_line
	HH_norm = spacial_freq_HH_norm*HH + spacial_freq_HH_line_norm*HH_line

	enhanced_image = pywt.idwt2((LL_norm, (LH_norm, HL_norm, HH_norm)), 'bior1.3')

	return ((enhanced_image/enhanced_image.max())*255).astype(np.uint8)