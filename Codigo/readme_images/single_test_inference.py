import torchvision
from PIL import Image
from inference import image_haze_removel
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToPILImage
import metric
import time
import evolution

def dehaze_test(image_path,weight):
	metrics = np.asarray([0,0])
	for input in image_path:
		hazy_input_image = Image.open(input)

		dehaze_image = image_haze_removel(hazy_input_image,weight)
		# torchvision.utils.save_image(dehaze_image, "dehaze.jpg")
		tensor_image = dehaze_image  # Your tensor image
		tensor_image = tensor_image.cpu()  # Move tensor to CPU if it's on GPU
		tensor_image = tensor_image.squeeze(0)  # Remove the batch dimension if present

		# Convert the tensor to a PIL image
		to_pil = ToPILImage()
		pil_image = to_pil(tensor_image)

		# hazy_input_image.show()
		# pil_image.show()

		# start_time = time.time()

		# print(metric.EME(np.array(pil_image),10,10)/metric.EME(np.array(hazy_input_image),10,10))
		# print(metric.IEM(np.array(hazy_input_image),np.array(pil_image)))
		# print(metric.PSNR(hazy_input_image,pil_image))

		# end_time = time.time()

		# print("Metric execution time: ", end_time - start_time , "seconds")
		# print(metric.SSIM(np.array(hazy_input_image),np.array(pil_image)))
		metrics = metrics + np.asarray([metric.PSNR(hazy_input_image,pil_image),metric.SSIM(np.array(hazy_input_image),np.array(pil_image))])
	return metrics/len(image_path)

path = "C:\\Users\\carlo\\Documents\\GitHub\\Mestrado\\Codigo\\img_teste\\"

img = path+ "047.jpg"

weight = torch.load('trained_weights/trained_LDNet.pth')
# mutated_weight = evolution.mutation(weight,0.1)
# child = evolution.crossing(weight,mutated_weight,0)

# single_dehaze_test(img,weight)
# single_dehaze_test(img,mutated_weight)
# single_dehaze_test(img,child)

# population = [weight,mutated_weight,child]
device1 = [	path+ "017.jpg",path+ "018.jpg",path+ "019.jpg",path+ "020.jpg",path+ "021.jpg",path+ "022.jpg",path+ "023.jpg",path+ "024.jpg"]
device2 = [	path+ "001.jpg", path+ "002.jpg",path+ "003.jpg",path+ "004.jpg",path+ "005.jpg",path+ "006.jpg",path+ "007.jpg",path+ "008.jpg",
			path+ "009.jpg",path+ "010.jpg",path+ "011.jpg",path+ "012.jpg",path+ "013.jpg",path+ "014.jpg",path+ "015.jpg",path+ "016.jpg"]
device3 = [	path+ "025.jpg", path+ "026.jpg",path+ "027.jpg",path+ "028.jpg",path+ "029.jpg",path+ "030.jpg"]

# device1 = [path+ "017.jpg",path+ "018.jpg",path+ "019.jpg"]

# device2 = [path+ "027.jpg",path+ "028.jpg"]

# device3 = [path+ "001.jpg"]

# fitness_score_each_device = [evolution.fitness_test(population,img_path_list1),evolution.fitness_test(population,img_path_list2),evolution.fitness_test(population,img_path_list3)]
# number_of_imgs_each_device = [1,3,2]

# print(evolution.aggregate_fitness_score(fitness_score_each_device,number_of_imgs_each_device))

evolutionProcess = evolution.EvolutionaryProcess(weight,25,5,0.01,3,[device1,device2,device3])

evolutionProcess.evolve()

# optimized_weight = evolutionProcess.return_apex()

# print("Original model: ")
# print(dehaze_test(device1+device2+device3 , weight))

# print("\nOptimized model: ")
# print(dehaze_test(device1+device2+device3 , optimized_weight))