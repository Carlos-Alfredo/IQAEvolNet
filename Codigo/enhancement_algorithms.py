import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from huggingface_hub import from_pretrained_keras
from transformers import AutoModel
from PIL import Image
import tensorflow as tf
from tabulate import tabulate
import matplotlib.pyplot as plt
from collections import OrderedDict

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

	for i in range(m):
		for j in range(n):
			image[i][j] = hist_eq[img_hef[i][j]]

	return image.astype(np.uint8)

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
