from huggingface_hub import from_pretrained_keras
from transformers import AutoModel
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
from enhancement_algorithms import UM, HEF,CLAHE, maxim_model, infer, imshow

def teste_inicial():
	image_path = "C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\Codigo\\x-ray-images-enhancement-master\\images\\001.jpg"
	#image_path = image_path = "C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\Codigo\\Ultrassom\\Origem\\MPX1005_synpic27455.png"
	#image_path = "C:\\Users\\carlo\\Documents\\GitHub\\ProjetoPecem\\Carlos-SVMClassifier\\dataset_pecem\\Ruim\\Imagem7.jpg"

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
	cl1 = CLAHE(img,2,8)
	end = time.time()
	print("Tempo de execução CLAHE =",end-start)

	cv2.imshow(input_image,"Input")
	cv2.imshow(final_pred_image,"Maxim")
	cv2.imshow(cl1,"CLAHE")
	cv2.waitKey(0)
	cv2.destroyallwindows()


	plt.figure(figsize=(15, 15))

	plt.subplot(1, 3, 1)
	input_image = np.asarray(Image.open(image_path).convert("RGB"), np.float32) / 255.0

	cv2.imshow(input_image,"Input")
	cv2.imshow(final_pred_image,"Maxim")
	cv2.imshow(cl1,"CLAHE")
	cv2.waitKey(0)
	cv2.destroyallwindows()

	
	imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY), "Input Image")

	plt.subplot(1, 3, 2)
	imshow(cv2.cvtColor(final_pred_image, cv2.COLOR_BGR2GRAY), "Predicted Image")

	plt.subplot(1, 3, 3)
	plt.imshow(cl1,cmap='gray')
	plt.title("CLAHE Image")

	plt.show()

def teste_geral():
	image_path = []
	for root, dirs, files in os.walk("C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\Codigo\\img_teste"):
		for file in files:
			image_path.append([os.path.join(root,file),file])
	print(len(image_path))
	maxim_time = []
	clahe_time = []
	maxim_metrics = [[],[],[],[],[],[],[]]
	clahe_metrics = [[],[],[],[],[],[],[]]
	model = maxim_model()
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	for path in image_path:
		start = time.time()
		final_pred_image = infer(path[0],model)
		final_pred_image = cv2.cvtColor(infer(path[0],model), cv2.COLOR_BGR2GRAY)*255
		end = time.time()
		maxim_time.append(end-start)

		img = cv2.imread(path[0], cv2.IMREAD_GRAYSCALE)

		start = time.time()
		cl1 = clahe.apply(img)
		end = time.time()
		clahe_time.append(end-start)

		cv2.imwrite("C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\\\Codigo\\img_resultado\\Clahe"+"\\"+path[1], cl1)

		cv2.imwrite("C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\\\Codigo\\img_resultado\\Maxim"+"\\"+path[1], final_pred_image)

		maxim_metrics[0].append(metric.MSE(img,final_pred_image))

		maxim_metrics[1].append(metric.EME(final_pred_image,10,10))

		maxim_metrics[2].append(metric.EMEE(final_pred_image,10,10))

		maxim_metrics[3].append(metric.RMSE(img,final_pred_image))

		maxim_metrics[4].append(metric.IEM(img,final_pred_image))

		maxim_metrics[5].append(metric.AG(final_pred_image))

		maxim_metrics[6].append(metric.SSIM(img,final_pred_image))

		clahe_metrics[0].append(metric.MSE(img,cl1))

		clahe_metrics[1].append(metric.EME(cl1,10,10))

		clahe_metrics[2].append(metric.EMEE(cl1,10,10))

		clahe_metrics[3].append(metric.RMSE(img,cl1))

		clahe_metrics[4].append(metric.IEM(img,cl1))

		clahe_metrics[5].append(metric.AG(cl1))

		clahe_metrics[6].append(metric.SSIM(img,cl1))



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

teste_inicial()