
# coding: utf-8

# In[5]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
import time
from PIL import Image
import skfuzzy as fuzz
import math
import timeit
import cv2
import tensorflow as tf
    
def bpdfhe(image):
    
    opimg = np.zeros_like(image)
 
    array1 = np.zeros(256*4+1, dtype = int)
    hist = array1[1:257]
    deltahist = array1[257:513]
    delta2hist = array1[513:769]
    histmax = array1[769:]
    

    hist, bin_edges = np.histogram(image, bins=256)

    x_qual = np.arange(0, 11, 1)

    membership= fuzz.trimf(x_qual, [0, 5, 10])

    fuzzyhist = np.zeros((np.size(hist)+np.size(membership)-1,1)).T;


    for counter in range(np.size(membership)):

        fuzzyhist = fuzzyhist + membership[counter]*np.concatenate((np.zeros(counter+1-1), hist, np.zeros(np.size(membership)-counter-1)),axis=0)

    fuzzyhist = fuzzyhist.T[math.ceil(np.size(membership)/2):-math.floor(np.size(membership)/2)+1]

    for i in range(255):
        deltahist[i] = (fuzzyhist[i+1]-fuzzyhist[i-1])/2
        delta2hist[i] = fuzzyhist[i+1]-2*fuzzyhist[i]+fuzzyhist[i-1]
    for i in range(255):
        if (deltahist[i+1]*deltahist[i-1]<0 and delta2hist[i]<0):
            histmax[i] = 1

    parts = np.where(histmax)[0]
    x = np.append(0,parts)
    x = np.append(x,255)

    span = np.zeros(x.shape[0], dtype = float) 
    M = np.zeros(x.shape[0], dtype = float) 
    factor = np.zeros(x.shape[0], dtype = float) 
    rang = np.zeros(x.shape[0], dtype = float) 
    start = np.zeros(x.shape[0], dtype = float) 

    Msum = np.cumsum(hist)

    for i in range(x.shape[0]):
        span[i-1]   = x[i] - x[i-1]
        if (span[i-1]<0):
            span[i-1]=1
        M[i-1] = Msum[x[i]]-Msum[x[i-1]]
        if (M[i-1]<0):
            M[i-1]=1
        factor[i-1] = span[i-1]*math.log10(M[i-1])

    factorsum = sum(factor)

    for i in range(x.shape[0]):    
        rang[i-1] = 255*factor[i-1]/factorsum

    start = np.cumsum(rang)
    start = np.append(0,start)
    start = start.astype(int)

    opimg = np.zeros_like(image)
    img2 = np.zeros_like(image)
    small = np.amin(image)
    big = np.amax(image)

    y = np.zeros(hist.shape, dtype = float) 
    for i in range(start.shape[0]):

        y[start[i-1]:start[i]] = start[i-1]+rang[i-1]/M[i-1]*np.cumsum(hist[start[i-1]:start[i]])


    opimg = y[image]

    return opimg

#image_path = "C:\\Users\\lucas vitoriano\\Desktop\\Mestrado\\Codigo\\img_teste\\001.jpg"
image_path = "C:\\Users\\lucas vitoriano\\Desktop\\Mestrado\\Codigo\\Ultrassom\\Origem\\MPX1005_synpic27455.png"
image = Image.open(image_path)
image = np.array(image)

enh_image = bpdfhe(image)

plt.figure(figsize=(15, 15))

plt.subplot(1, 2, 1)
plt.imshow(image,cmap="gray")
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(enh_image.astype(int),cmap="gray")
plt.title( "Predicted Image")

plt.show()



