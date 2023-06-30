import metric
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToPILImage

def list_split(original_list,number_of_splits):
	sub_list = []
	split = []
	number_of_elements = len(original_list)
	split_size = int(number_of_elements/number_of_splits)
	extra_elements = number_of_elements%number_of_splits
	last_element = 0
	for i in range(0,number_of_splits):
		if i < extra_elements:
			sub_list.append(original_list[last_element:last_element+split_size+1])
			last_element = last_element+split_size+1
		else:
			sub_list.append(original_list[last_element:last_element+split_size])
			last_element = last_element+split_size
	return sub_list

def metric_calculation(image_pair_list):
	# tensor_image = enhanced_image  # Your tensor image
	# tensor_image = tensor_image.cpu()  # Move tensor to CPU if it's on GPU
	# tensor_image = tensor_image.squeeze(0)  # Remove the batch dimension if present

	# # Convert the tensor to a PIL image
	# to_pil = ToPILImage()
	# pil_image = to_pil(tensor_image)

	# img_original = original_image
	# img_original = img_original.cpu()
	# img_original = img_original.squeeze(0)
	# pil_original = to_pil(img_original)

	# fitness_measure.append([metric.EME(np.array(pil_image),10,10)/metric.EME(np.array(pil_original),10,10),metric.SSIM(np.array(pil_original),np.array(pil_image))])
	fitness_measure_list = []
	print(type(image_pair_list))
	for image_pair in image_pair_list:
		enhanced_image = image_pair[0]
		original_image = image_pair[1]
		print(np.array(original_image).shape)
		fitness_measure = [metric.PSNR(np.array(original_image),np.array(enhanced_image)),metric.SSIM(np.array(original_image),np.array(enhanced_image))]
		fitness_measure_list.append(fitness_measure)
	return fitness_measure_list