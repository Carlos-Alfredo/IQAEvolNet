from huggingface_hub import from_pretrained_keras
from PIL import Image

import tensorflow as tf
import numpy as np
import requests
import matplotlib.pyplot as plt
import cv2
import time
import os
import metric
from tabulate import tabulate

def mod_padding_symmetric(image, factor=64):
    """Padding the image to be divided by factor."""
    height, width = image.shape[0], image.shape[1]
    height_pad, width_pad = ((height + factor) // factor) * factor, (
        (width + factor) // factor
    ) * factor
    padh = height_pad - height if height % factor != 0 else 0
    padw = width_pad - width if width % factor != 0 else 0
    image = tf.pad(
        image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)], mode="REFLECT"
    )
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
    return new_model

def teste_inicial():
	image_path = "C:\\Users\\carlo\\Documents\\Mestrado\\Codigo\\x-ray-images-enhancement-master\\images\\001.jpg"

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

	start = time.time()
	model = maxim_model()
	end = time.time()
	print("Tempo de carregamento do modelo = ",end-start)

	start = time.time()
	final_pred_image = infer(image_path,model)
	end = time.time()
	print("Tempo de execução Maxim =",end-start)

	img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

	start = time.time()
	cl1 = clahe.apply(img)
	end = time.time()
	print("Tempo de execução CLAHE =",end-start)

	plt.figure(figsize=(15, 15))

	plt.subplot(1, 3, 1)
	input_image = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0
	imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), "Input Image")

	plt.subplot(1, 3, 2)
	imshow(cv2.cvtColor(final_pred_image, cv2.COLOR_BGR2GRAY), "Predicted Image")

	plt.subplot(1, 3, 3)
	plt.imshow(cl1,cmap='gray')
	plt.title("CLAHE Image")

	plt.show()

def teste_geral():
	image_path = []
	for root, dirs, files in os.walk("C:\\Users\\carlo\\Documents\\Mestrado\\Codigo\\img_teste"):
		for file in files:
			image_path.append([os.path.join(root,file),file])
	print(image_path[0])
	maxim_time = []
	clahe_time = []
	maxim_metrics = [[],[],[],[],[],[]]
	clahe_metrics = [[],[],[],[],[],[]]
	model = maxim_model()
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	for path in image_path:
		start = time.time()
		#final_pred_image = infer(path[0],model)
		final_pred_image = cv2.cvtColor(infer(path[0],model), cv2.COLOR_BGR2GRAY)*255
		end = time.time()
		maxim_time.append(end-start)

		img = cv2.imread(path[0], cv2.IMREAD_GRAYSCALE)

		start = time.time()
		cl1 = clahe.apply(img)
		end = time.time()
		clahe_time.append(end-start)
		print(img.shape)
		print(final_pred_image.shape)
		print(cl1.shape)

		cv2.imwrite("C:\\Users\\carlo\\Documents\\Mestrado\\Codigo\\img_resultado\\Clahe"+"\\"+path[1], cl1)

		cv2.imwrite("C:\\Users\\carlo\\Documents\\Mestrado\\Codigo\\img_resultado\\Maxim"+"\\"+path[1], final_pred_image)

		maxim_metrics[0].append(metric.MSE(img,final_pred_image))

		maxim_metrics[1].append(metric.EME(final_pred_image,10,10))

		maxim_metrics[2].append(metric.EMEE(final_pred_image,10,10))

		maxim_metrics[3].append(metric.RMSE(img,final_pred_image))

		maxim_metrics[4].append(metric.IEM(img,final_pred_image))

		maxim_metrics[5].append(metric.AG(final_pred_image))

		clahe_metrics[0].append(metric.MSE(img,cl1))

		clahe_metrics[1].append(metric.EME(cl1,10,10))

		clahe_metrics[2].append(metric.EMEE(cl1,10,10))

		clahe_metrics[3].append(metric.RMSE(img,cl1))

		clahe_metrics[4].append(metric.IEM(img,cl1))

		clahe_metrics[5].append(metric.AG(cl1))

	tabela_maxim=np.zeros((2,6))
	tabela_clahe=np.zeros((2,6))
	for i in range(0,6):
		tabela_maxim[0,i] = np.mean(maxim_metrics[i])
		tabela_maxim[1,i] = np.std(maxim_metrics[i])

	tabela_clahe=np.zeros((2,6))
	for i in range(0,6):
		tabela_clahe[0,i] = np.mean(clahe_metrics[i])
		tabela_clahe[1,i] = np.std(clahe_metrics[i])

	print("Maxim mean time = ", np.mean(maxim_time))
	print("Maxim std time = ", np.std(maxim_time))
	print("Clahe mean time = ", np.mean(clahe_time))
	print("Clahe std time = ", np.std(clahe_time))

	np.savetxt("maxim_statistics.csv",tabela_maxim,delimiter=",")
	np.savetxt("clahe_statistics.csv",tabela_clahe,delimiter=",")
	
	print(tabulate(tabela_maxim))
	print(tabulate(tabela_clahe))

teste_geral()