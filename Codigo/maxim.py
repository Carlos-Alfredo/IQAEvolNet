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

	img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	model = maxim_model()
	start = time.time()
	#final_pred_image = infer(path[0],model)
	maxim_img = cv2.cvtColor(infer(image_path,model), cv2.COLOR_BGR2GRAY)*255
	end = time.time()
	print("Tempo de execução Maxim =",end-start)

	start = time.time()
	clahe_img = CLAHE(img,2,8)
	end = time.time()
	print("Tempo de execução CLAHE =",end-start)

	start = time.time()
	um_img = UM(img,1,0)
	end = time.time()
	print("Tempo de execução UM =",end-start)

	start = time.time()
	hef_image = HEF(img,70)
	end = time.time()
	print("Tempo de execução HEF =",end-start)

	# start = time.time()
	# model = maxim_model()
	# end = time.time()
	# print("Tempo de carregamento do modelo = ",end-start)

	# start = time.time()
	# final_pred_image = infer(image_path,model)
	# end = time.time()
	# print("Tempo de execução Maxim =",end-start)

	# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

	# start = time.time()
	# cl1 = CLAHE(img,2,8)
	# end = time.time()
	# print("Tempo de execução CLAHE =",end-start)

	plt.figure(figsize=(15, 15))
	img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	plt.subplot(2, 3, 1)
	imshow(img, "Input Image")

	plt.subplot(2, 3, 2)
	#imshow(cv2.cvtColor(final_pred_image, cv2.COLOR_BGR2GRAY), "Predicted Image")
	imshow(maxim_img, "Maxim Image")

	plt.subplot(2, 3, 3)
	imshow(clahe_img, "CLAHE Image")
	
	plt.subplot(2, 3, 4)
	imshow(um_img, "UM Image")

	plt.subplot(2, 3, 5)
	imshow(hef_image, "HEF Image")

	plt.show()

def teste_geral():
	image_path = []
	for root, dirs, files in os.walk("C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\Codigo\\img_teste"):
		for file in files:
			image_path.append([os.path.join(root,file),file])
	tot = len(image_path)
	curr = 0
	maxim_time = []
	clahe_time = []
	um_time    = []
	hef_time   = []
	maxim_metrics = [[],[],[],[],[],[],[]]
	clahe_metrics = [[],[],[],[],[],[],[]]
	um_metrics    = [[],[],[],[],[],[],[]]
	hef_metrics   = [[],[],[],[],[],[],[]]
	model = maxim_model()
	#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	for path in image_path:
		print(str(curr)+"/"+str(tot))
		curr = curr + 1
		start = time.time()
		#final_pred_image = infer(path[0],model)
		maxim_img = cv2.cvtColor(infer(path[0],model), cv2.COLOR_BGR2GRAY)*255
		end = time.time()
		maxim_time.append(end-start)

		img = cv2.imread(path[0], cv2.IMREAD_GRAYSCALE)

		start = time.time()
		clahe_img = CLAHE(img,2,8)
		end = time.time()
		clahe_time.append(end-start)

		start = time.time()
		um_img = UM(img,1,0)
		end = time.time()
		um_time.append(end-start)

		start = time.time()
		hef_img = HEF(img,70)
		end = time.time()
		hef_time.append(end-start)

		img = cv2.imread(path[0], cv2.IMREAD_GRAYSCALE)

		cv2.imwrite("C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\\\Codigo\\img_resultado\\Maxim"+"\\"+path[1], maxim_img)

		cv2.imwrite("C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\\\Codigo\\img_resultado\\Clahe"+"\\"+path[1], clahe_img)

		cv2.imwrite("C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\\\Codigo\\img_resultado\\UM"+"\\"+path[1], um_img)

		cv2.imwrite("C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\\\Codigo\\img_resultado\\HEF"+"\\"+path[1], hef_img)

		# Maxim Metrics
		maxim_metrics[0].append(metric.MSE(img,maxim_img))

		maxim_metrics[1].append(metric.EME(maxim_img,10,10))

		maxim_metrics[2].append(metric.EMEE(maxim_img,10,10))

		maxim_metrics[3].append(metric.RMSE(img,maxim_img))

		maxim_metrics[4].append(metric.IEM(img,maxim_img))

		maxim_metrics[5].append(metric.AG(maxim_img))

		maxim_metrics[6].append(metric.SSIM(img,maxim_img))

		# CLAHE Metrics
		clahe_metrics[0].append(metric.MSE(img,clahe_img))

		clahe_metrics[1].append(metric.EME(clahe_img,10,10))

		clahe_metrics[2].append(metric.EMEE(clahe_img,10,10))

		clahe_metrics[3].append(metric.RMSE(img,clahe_img))

		clahe_metrics[4].append(metric.IEM(img,clahe_img))

		clahe_metrics[5].append(metric.AG(clahe_img))

		clahe_metrics[6].append(metric.SSIM(img,clahe_img))

		#UM Metrics
		um_metrics[0].append(metric.MSE(img,um_img))

		um_metrics[1].append(metric.EME(um_img,10,10))

		um_metrics[2].append(metric.EMEE(um_img,10,10))

		um_metrics[3].append(metric.RMSE(img,um_img))

		um_metrics[4].append(metric.IEM(img,um_img))

		um_metrics[5].append(metric.AG(um_img))

		um_metrics[6].append(metric.SSIM(img,um_img))

		#HEF Metrics
		hef_metrics[0].append(metric.MSE(img,hef_img))

		hef_metrics[1].append(metric.EME(hef_img,10,10))

		hef_metrics[2].append(metric.EMEE(hef_img,10,10))

		hef_metrics[3].append(metric.RMSE(img,hef_img))

		hef_metrics[4].append(metric.IEM(img,hef_img))

		hef_metrics[5].append(metric.AG(hef_img))

		hef_metrics[6].append(metric.SSIM(img,hef_img))

	tabela_maxim=np.zeros((2,7))
	tabela_clahe=np.zeros((2,7))
	tabela_um=np.zeros((2,7))
	tabela_hef=np.zeros((2,7))
	tabela_time=np.zeros((2,4))
	for i in range(0,7):
		tabela_maxim[0,i] = np.mean(maxim_metrics[i])
		tabela_maxim[1,i] = np.std(maxim_metrics[i])

		tabela_clahe[0,i] = np.mean(clahe_metrics[i])
		tabela_clahe[1,i] = np.std(clahe_metrics[i])

		tabela_um[0,i] = np.mean(um_metrics[i])
		tabela_um[1,i] = np.std(um_metrics[i])

		tabela_hef[0,i] = np.mean(hef_metrics[i])
		tabela_hef[1,i] = np.std(hef_metrics[i])
	
	tabela_time[0,0] = np.mean(maxim_time)
	tabela_time[1,0] = np.std(maxim_time)
	tabela_time[0,1] = np.mean(clahe_time)
	tabela_time[1,1] = np.std(clahe_time)
	tabela_time[0,2] = np.mean(um_time)
	tabela_time[1,2] = np.std(um_time)
	tabela_time[0,3] = np.mean(hef_time)
	tabela_time[1,3] = np.std(hef_time)

	np.savetxt("maxim_statistics.csv",tabela_maxim,delimiter=",")
	np.savetxt("clahe_statistics.csv",tabela_clahe,delimiter=",")
	np.savetxt("um_statistics.csv",tabela_um,delimiter=",")
	np.savetxt("hef_statistics.csv",tabela_hef,delimiter=",")
	np.savetxt("time_statistics.csv",tabela_time,delimiter=",")

teste_geral()