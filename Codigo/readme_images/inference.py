# @author: hayat
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import image_data_loader
import lightdehazeNet
import numpy
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
from matplotlib import pyplot as plt
import cv2

def image_haze_removel(input_image,weight):

	hazy_image = (np.asarray(input_image)/255.0)

	hazy_image = torch.from_numpy(hazy_image).float()
	hazy_image = hazy_image.permute(2,0,1)
	hazy_image = hazy_image.cuda().unsqueeze(0)

	ld_net = lightdehazeNet.LightDehaze_Net().cuda()
	
	# ld_net.load_state_dict(torch.load('trained_weights/trained_LDNet.pth'))
	ld_net.load_state_dict(weight)
	# print((ld_net.state_dict()['e_conv_layer8.weight'])[2,35,:,:])
	# print(torch.std((ld_net.state_dict()['e_conv_layer8.weight'])[2,35,:,:]))
	# print(torch.randn_like((ld_net.state_dict()['e_conv_layer8.weight'])[2,35,:,:])*0.1 + 1)
	dehaze_image = ld_net(hazy_image)
	return dehaze_image

def image_haze_removal(input_image,ld_net):

	hazy_image = (np.asarray(input_image)/255.0)

	hazy_image = torch.from_numpy(hazy_image).float()
	hazy_image = hazy_image.permute(2,0,1)
	hazy_image = hazy_image.cuda().unsqueeze(0)

	dehaze_image = ld_net(hazy_image)
	
	return dehaze_image