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
from torchvision.transforms import ToPILImage
import importlib
importlib.reload(lightdehazeNet)

def image_haze_removal(input_image,ld_net):

	#hazy_image = (np.asarray(input_image)/255.0)

	hazy_image = torch.from_numpy(input_image)#.float()
	#hazy_image = hazy_image.permute(2,0,1)
	hazy_image = hazy_image.cuda().unsqueeze(0).unsqueeze(0)

	#hazy_image = ((hazy_image.unsqueeze(0)).permute(1,0,2,3)).cuda()

	dehaze_image_tensor = ld_net(hazy_image)
	dehaze_image_pil = ToPILImage()(dehaze_image_tensor.cpu().squeeze(0))
	del dehaze_image_tensor
	return dehaze_image_pil