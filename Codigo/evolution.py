import torch
import copy
from PIL import Image
from inference import image_haze_removel
import numpy as np
from torchvision.transforms import ToPILImage
import metric
import torch.multiprocessing as mp

class EvolutionaryProcess:
	def __init__(self, starting_weight,population_size,parent_size,chance_of_mutation,max_rounds_without_improvement,devices):
		self.round_count = 0
		self.population_size = population_size
		self.parent_size = parent_size
		self.chance_of_mutation = chance_of_mutation
		self.apex = starting_weight
		self.rounds_without_improvement = 0
		self.max_rounds_without_improvement = max_rounds_without_improvement
		self.devices = devices
		self.device_weight = []
		for img_path_list in devices:
			self.device_weight.append(len(img_path_list))
		self.population = [starting_weight]
		for i in range(1,population_size):
			self.population.append(self.mutation(starting_weight))

	def mutation(self,weight):
		dimensions = 	[weight['e_conv_layer8.weight'].size(dim=0),
						weight['e_conv_layer8.weight'].size(dim=1),
						weight['e_conv_layer8.weight'].size(dim=2),
						weight['e_conv_layer8.weight'].size(dim=3)]
		mutated_weight = copy.deepcopy(weight)

		for j in range(0,dimensions[1]):
			# At a rate of chance_of_mutation
			if(torch.rand(1)<self.chance_of_mutation):
				# Creates a random mutation tensor in the interval of [-1,1]
				mutation_tensor = torch.rand((dimensions[2],dimensions[3]),device='cpu') - torch.rand((dimensions[2],dimensions[3]),device='cpu')
				if torch.rand(1)<0.5:
					mutation_tensor = mutation_tensor.pow_(-1)
				# Mutates the tensor on all color channels
				for i in range(0,dimensions[0]):
					mutated_weight['e_conv_layer8.weight'][i,j,:,:] = weight['e_conv_layer8.weight'][i,j,:,:]*mutation_tensor

		return mutated_weight

	# parentA: weight
	# parentB: weight
	# chance of mutation: real number([0,1])
	# return: A new weight(child)

	def crossing(self,parentA,parentB):
		dimensions = 	[parentA['e_conv_layer8.weight'].size(dim=0),
						parentA['e_conv_layer8.weight'].size(dim=1),
						parentA['e_conv_layer8.weight'].size(dim=2),
						parentA['e_conv_layer8.weight'].size(dim=3)]
		
		child = copy.deepcopy(parentA)

		for i in range(0,dimensions[0]):
			for j in range(0,dimensions[1]):
				child['e_conv_layer8.weight'][i,j,:,:] = (parentA['e_conv_layer8.weight'][i,j,:,:] + parentB['e_conv_layer8.weight'][i,j,:,:])/2

		return self.mutation(child)



	# parents: List of weights that will originate the next generation
	# next_generation: List of weights representing the next generation

	# def mutation(self,weight):
	# 	for layer in weight.keys():
	# 		dimensions = 	[weight[layer].size(dim=0),
	# 						weight[layer].size(dim=1),
	# 						weight[layer].size(dim=2),
	# 						weight[layer].size(dim=3)]
	# 		mutated_weight = copy.deepcopy(weight)

	# 		for j in range(0,dimensions[1]):
	# 			# At a rate of chance_of_mutation
	# 			if(torch.rand(1)<self.chance_of_mutation):
	# 				# Creates a random mutation tensor in the interval of [-1,1]
	# 				mutation_tensor = torch.rand((dimensions[2],dimensions[3]),device='cuda:0') - torch.rand((dimensions[2],dimensions[3]),device='cuda:0')
	# 				# Mutates the tensor on all color channels
	# 				for i in range(0,dimensions[0]):
	# 					mutated_weight[layer][i,j,:,:] = weight[layer][i,j,:,:]*mutation_tensor

	# 		return mutated_weight

	# parentA: weight
	# parentB: weight
	# chance of mutation: real number([0,1])
	# return: A new weight(child)

	# def crossing(self,parentA,parentB):
	# 	for layer in parentA.keys():
	# 		dimensions = 	[parentA[layer].size(dim=0),
	# 						parentA[layer].size(dim=1),
	# 						parentA[layer].size(dim=2),
	# 						parentA[layer].size(dim=3)]
			
	# 		child = copy.deepcopy(parentA)

	# 		for i in range(0,dimensions[0]):
	# 			for j in range(0,dimensions[1]):
	# 				child[layer][i,j,:,:] = (parentA[layer][i,j,:,:] + parentB[layer][i,j,:,:])/2

	# 		return self.mutation(child)

	def next_generation(self,parents):
		next_generation = []
		for i in range(0,len(parents)):
			for j in range(0,len(parents)):
				if i!=j:
					next_generation.append(self.crossing(parents[i],parents[j]))
				else:
					next_generation.append(parents[i])
		return next_generation

	# population: List with the all the weights of a specific generation
	# img_path_list: list with the paths of the images that will be used for the test
	# returns: an array with the partial fitness score of each device ( population x number of metrics )

	def fitness_test(self,img_path_list):
		# cuda_device = torch.cuda.current_device()
		# torch.cuda.set_device(cuda_device)
		fitness_score = []
		max_eme = 0
		max_ssim = 0
		for i in range(0,len(self.population)):
			weight = self.population[i]
			fitness_measure = []
			for img_path in img_path_list:
				hazy_input_image = Image.open(img_path)
				dehaze_image = image_haze_removel(hazy_input_image,weight)
				# torchvision.utils.save_image(dehaze_image, "dehaze.jpg")
				tensor_image = dehaze_image  # Your tensor image
				tensor_image = tensor_image.cpu()  # Move tensor to CPU if it's on GPU
				tensor_image = tensor_image.squeeze(0)  # Remove the batch dimension if present

				# Convert the tensor to a PIL image
				to_pil = ToPILImage()
				pil_image = to_pil(tensor_image)

				# print(metric.EME(np.array(pil_image),10,10)/metric.EME(np.array(hazy_input_image),10,10))
				# print(metric.IEM(np.array(hazy_input_image),np.array(pil_image)))
				# print(metric.PSNR(hazy_input_image,pil_image))
				# print(metric.SSIM(np.array(hazy_input_image),np.array(pil_image)))
				fitness_measure.append([metric.EME(np.array(pil_image),10,10)/metric.EME(np.array(hazy_input_image),10,10),metric.SSIM(np.array(hazy_input_image),np.array(pil_image))])

			fitness_measure = np.asarray(fitness_measure)

			fitness_measure_avg = np.mean(fitness_measure,axis=0)

			fitness_score.append(fitness_measure_avg)

		return np.asarray(fitness_score)

	# fitness_score_each_device: List with the fitness scores matrix(population_size x number of metrics) of each device
	# number_of_imgs_each_device: List with the number of images in each device
	# returns: numpy vector( population_size ) with the fitness score total of each individual
	def aggregate_fitness_score(self,fitness_score_each_device,number_of_imgs_each_device):
		fitness_score_aggregated = fitness_score_each_device[0]*number_of_imgs_each_device[0]
		for i in range(1,len(fitness_score_each_device)):
			fitness_score_aggregated = fitness_score_aggregated + fitness_score_each_device[i]*number_of_imgs_each_device[i]
		fitness_score_aggregated = fitness_score_aggregated/sum(number_of_imgs_each_device)
		fitness_score_normalized = fitness_score_aggregated/np.max(fitness_score_aggregated,axis=0)

		fitness_score_total = fitness_score_normalized[:,0]*fitness_score_normalized[:,1]
		return fitness_score_total

	def evolve(self,num_processes):
		args_list = []  # List of arguments for the task function
		for device in self.devices:
			args_list.append(device)
		print(args_list)
		if self.round_count >20:
			return 1
		print("-> Round ",self.round_count)
		self.round_count += 1
		fitness_score_each_device = []
		# for img_path_list in self.devices:
			# fitness_score_each_device.append(self.fitness_test(self.population,img_path_list))
		# num_processes = torch.cuda.device_count()  # Get the number of available CUDA devices
		# pool = mp.Pool(processes=num_processes)
		# num_processes = mp.cpu_count()  # Get the number of available CPU cores
		print(len(args_list))
		pool = mp.Pool(processes=num_processes)
		results = pool.map(self.fitness_test, args_list)
		pool.close()
		pool.join()
		fitness_score_each_device = results
		fitness_score_total = self.aggregate_fitness_score(fitness_score_each_device,self.device_weight)
		population_ranking = np.argsort(fitness_score_total)[::-1]
		equal = all(torch.allclose(self.apex[key], self.population[population_ranking[0]][key]) for key in self.apex.keys())
		if equal:
			self.rounds_without_improvement += 1
			print("There was no improvement this round!")
		else:
			self.apex = self.population[population_ranking[0]]
			self.rounds_without_improvement = 0
			print("New apex detected!")
		# if self.rounds_without_improvement >= self.max_rounds_without_improvement:
			# return 1
		# else:
		surviving_population = []
		for i in range(0,self.parent_size):
			surviving_population.append(self.population[population_ranking[i]])
		self.population = self.next_generation(surviving_population)
			# self.evolve()

	def return_apex(self):
		return self.apex