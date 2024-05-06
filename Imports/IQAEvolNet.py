import torch
import copy
from PIL import Image
from inference import image_haze_removal
import inference
import numpy as np
from torchvision.transforms import ToPILImage
import lightdehazeNet
import time
from skimage import img_as_ubyte
import cv2
import tensorflow
import pyiqa
from metric import EME,AME

class MetricDeclaration:

	def __init__(self,niqe=0,clipiqa=0,maniqa=0,hyperiqa=0,musiq=0,paq2piq=0,brisque=0,pi=0,nrqm=0,ilniqe=0,entropy=0,psnr=0,ssim=0,eme=0,ame=0):
		self.niqe = -niqe #Lower niqe means better quality
		self.clipiqa = clipiqa
		self.maniqa = maniqa
		self.hyperiqa = hyperiqa
		self.musiq = musiq
		self.paq2piq = paq2piq
		self.brisque = -brisque #Lower brisque means better quality
		self.pi = -pi #Lower brisque means better quality
		self.nrqm = nrqm
		self.ilniqe = -ilniqe #Lower ilniqe means better quality
		self.entropy = entropy
		self.psnr = psnr
		self.ssim = ssim
		self.eme = eme
		self.ame = ame

class IQAEvolNet:

	def __init__(self, ld_net, starting_population,chance_of_mutation,
		noise_intensity,noise_decay,data_loader,batch_size_per_round,
		apex_threshold,metric_declaration):
		self.epoch_count = 0
		self.ld_net = ld_net
		self.apex_fitness_measure_avg = 0
		self.parent_size = int(len(starting_population) ** 0.5)
		self.population_size = self.parent_size ** 2
		self.chance_of_mutation = chance_of_mutation
		self.apex = None#baseline_weight['e_conv_layer8.weight']
		self.apex_score = None
		self.apex_timeline = []
		self.epochs_without_improvement = 0
		self.noise_intensity = noise_intensity
		self.noise_decay = noise_decay
		self.population = starting_population[(len(starting_population)-self.population_size):]
		self.data_loader = data_loader
		self.batch_size_per_round = batch_size_per_round
		self.apex_threshold = apex_threshold
		self.metric = metric_declaration
		self.metric_weights = np.asarray([	self.metric.niqe, self.metric.clipiqa,
											self.metric.maniqa, self.metric.hyperiqa,
											self.metric.musiq, self.metric.paq2piq,
											self.metric.brisque, self.metric.pi,
											self.metric.nrqm, self.metric.ilniqe,
											self.metric.entropy, self.metric.ssim, self.metric.psnr,
											self.metric.eme,self.metric.ame])

	def mutation(self,weight):
		dimensions = weight.size()
		mutated_weight = weight.clone().detach()

		if len(list(dimensions)) == 4:
			for j in range(0,dimensions[1]):
				# At a rate of chance_of_mutation * (1+rounds_without_improvement)
				if(torch.rand(1)<self.chance_of_mutation):
					# Creates a mutation sensor from a normal distribution
					mutation_tensor = torch.normal(mean=1,std=self.noise_intensity,size=dimensions[2:],device='cuda')# + torch.ones(dimensions[2:],dtype=torch.float32,device='cuda')
					# Mutates the tensor on all color channels
					for i in range(0,dimensions[0]):
						mutated_weight[i,j] = weight[i,j]*mutation_tensor
		if len(list(dimensions)) == 1:
			if(torch.rand(1)<self.chance_of_mutation):
				# Creates a mutation sensor from a normal distribution
				mutation_tensor = torch.normal(mean=1,std=self.noise_intensity,size=dimensions,device='cuda')# + torch.ones(dimensions,dtype=torch.float32,device='cuda')
				# Mutates the tensor on all color channels
				mutated_weight = weight*mutation_tensor
		return mutated_weight

	# parentA: dict representing the parameters of a CNN
	# parentB: dict representing the parameters of a CNN
	# return: A new dict(child)

	def crossing(self,parentA,parentB):
		keys = list(parentA.keys())
		child = copy.deepcopy(parentA)
		crossing_factor = np.random.normal(0.5,0.5/3)
		for key in keys:
			weight1 = parentA[key]
			weight2 = parentB[key]
			#weight_child = weight1
			weight_child = weight1*crossing_factor + weight2*(1-crossing_factor)
			if key.endswith('weight') or key.endswith('bias'):
				child[key] = self.mutation(weight_child)
			else:
				child[key] = weight_child

		return child

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

	def fitness_test(self, index):
		fitness_score = []
		ld_net = self.ld_net
		to_pil = ToPILImage(mode=None)
		image_batch,label_batch = self.data_loader.get_item(index, mode='train')
		device = torch.device("cuda")
		if self.metric.niqe!= 0:
			iqa_niqe = pyiqa.create_metric('niqe', device=device)
		if self.metric.clipiqa!= 0:
			iqa_clipiqa = pyiqa.create_metric('clipiqa', device=device)
		if self.metric.maniqa!= 0:
			iqa_maniqa = pyiqa.create_metric('maniqa', device=device)
		if self.metric.hyperiqa!= 0:
			iqa_hyperiqa = pyiqa.create_metric('hyperiqa', device=device)
		if self.metric.musiq!= 0:
			iqa_musiq = pyiqa.create_metric('musiq', device=device)
		if self.metric.paq2piq!= 0:
			iqa_paq2piq = pyiqa.create_metric('paq2piq', device=device)
		if self.metric.brisque!= 0:
			iqa_brisque = pyiqa.create_metric('brisque', device=device)
		if self.metric.pi!= 0:
			iqa_pi = pyiqa.create_metric('pi', device=device)
		if self.metric.nrqm!= 0:
			iqa_nrqm = pyiqa.create_metric('nrqm', device=device)
		if self.metric.ilniqe!= 0:
			iqa_ilniqe = pyiqa.create_metric('ilniqe', device=device)
		if self.metric.entropy!= 0:
			iqa_entropy = pyiqa.create_metric('entropy', device=device)
		if self.metric.ssim != 0:
			iqa_ssim = pyiqa.create_metric('ssim', device=device)
		if self.metric.psnr != 0:
			iqa_psnr = pyiqa.create_metric('psnr', device=device)
		for i in range(0,self.population_size):
			weight = self.population[i]
			ld_net.load_state_dict(weight,strict=True)
			fitness_measure_list = []
			for i in range(0,image_batch.shape[0]):
				image = image_batch[i]
				dehaze_image = inference.image_haze_removal(image,ld_net)
				enhanced_image = dehaze_image.convert('RGB')
				original_image = to_pil(image_batch[i]*255).convert('RGB')
				if self.metric.niqe != 0:
					niqe = iqa_niqe(enhanced_image).item()
				else:
					niqe = 1
				if self.metric.clipiqa != 0:
					clipiqa = iqa_clipiqa(enhanced_image).item()
				else:
					clipiqa = 1
				if self.metric.maniqa != 0:
					maniqa = iqa_maniqa(enhanced_image).item()
				else:
					maniqa = 1
				if self.metric.hyperiqa != 0:
					hyperiqa = iqa_hyperiqa(enhanced_image).item()
				else:
					hyperiqa = 1
				if self.metric.musiq != 0:
					musiq = iqa_musiq(enhanced_image).item()
				else:
					musiq = 1
				if self.metric.paq2piq != 0:
					paq2piq = iqa_paq2piq(enhanced_image).item()
				else:
					paq2piq = 1
				if self.metric.brisque != 0:
					brisque = iqa_brisque(enhanced_image).item()
				else:
					brisque = 1
				if self.metric.pi != 0:
					pi = iqa_pi(enhanced_image).item()
				else:
					pi = 1
				if self.metric.nrqm != 0:
					nrqm = iqa_nrqm(enhanced_image).item()
				else:
					nrqm = 1
				if self.metric.ilniqe != 0:
					ilniqe = iqa_ilniqe(enhanced_image).item()
				else:
					ilniqe = 1
				if self.metric.entropy != 0:
					entropy = iqa_entropy(enhanced_image).item()
				else:
					entropy = 1
				if self.metric.ssim != 0:
					ssim = iqa_ssim(original_image,enhanced_image).item()
				else:
					ssim = 1
				if self.metric.psnr != 0:
					psnr = iqa_psnr(original_image,enhanced_image).item()
				else:
					psnr = 1
				if self.metric.eme != 0:
					eme = EME(enhanced_image,8,8)
				else:
					eme = 1
				if self.metric.ame != 0:
					ame = AME(enhanced_image,8,8)
				else:
					ame = 1
				fitness_measure = np.asarray([niqe, clipiqa,
											maniqa, hyperiqa,
											musiq, paq2piq,
											brisque, pi,
											nrqm, ilniqe,
											entropy, ssim, psnr,
											eme, ame])
				fitness_measure_list.append(fitness_measure)
			fitness_measure = np.asarray(fitness_measure_list)
			fitness_measure_avg = np.mean(fitness_measure,axis=0)
			fitness_score.append(fitness_measure_avg)
		return np.asarray(fitness_score)

	# fitness_score_each_device: List with the fitness scores matrix(population_size x number of metrics) of each device
	# number_of_imgs_each_device: List with the number of images in each device
	# returns: numpy vector( population_size ) with the fitness score total of each individual
	def aggregate_fitness_score(self,fitness_score):
		fitness_score_total = 1
		for i in range(0,fitness_score.shape[1]):
			fitness_score_total = fitness_score_total*(fitness_score[:,i] ** self.metric_weights[i])
		return fitness_score_total

	def full_dataset_test(self, individual, mode='train'):
		fitness_score = []
		ld_net = self.ld_net
		to_pil = ToPILImage(mode=None)
		weight = individual
		ld_net.load_state_dict(weight,strict=True)
		device = torch.device("cuda")
		if self.metric.niqe!= 0:
			iqa_niqe = pyiqa.create_metric('niqe', device=device)
		if self.metric.clipiqa!= 0:
			iqa_clipiqa = pyiqa.create_metric('clipiqa', device=device)
		if self.metric.maniqa!= 0:
			iqa_maniqa = pyiqa.create_metric('maniqa', device=device)
		if self.metric.hyperiqa!= 0:
			iqa_hyperiqa = pyiqa.create_metric('hyperiqa', device=device)
		if self.metric.musiq!= 0:
			iqa_musiq = pyiqa.create_metric('musiq', device=device)
		if self.metric.paq2piq!= 0:
			iqa_paq2piq = pyiqa.create_metric('paq2piq', device=device)
		if self.metric.brisque!= 0:
			iqa_brisque = pyiqa.create_metric('brisque', device=device)
		if self.metric.pi!= 0:
			iqa_pi = pyiqa.create_metric('pi', device=device)
		if self.metric.nrqm!= 0:
			iqa_nrqm = pyiqa.create_metric('nrqm', device=device)
		if self.metric.ilniqe!= 0:
			iqa_ilniqe = pyiqa.create_metric('ilniqe', device=device)
		if self.metric.entropy!= 0:
			iqa_entropy = pyiqa.create_metric('entropy', device=device)
		if self.metric.ssim != 0:
			iqa_ssim = pyiqa.create_metric('ssim', device=device)
		if self.metric.psnr != 0:
			iqa_psnr = pyiqa.create_metric('psnr', device=device)
		fitness_measure_list = []
		for i in range(0,self.data_loader.__len__(mode=mode)):
			image_batch,label_batch = self.data_loader.get_item(i, mode=mode)
			for i in range(0,image_batch.shape[0]):
				image = image_batch[i]
				dehaze_image = inference.image_haze_removal(image,ld_net)
				enhanced_image = dehaze_image.convert('RGB')
				original_image = to_pil(image_batch[i]*255).convert('RGB')
				if self.metric.niqe != 0:
					niqe = iqa_niqe(enhanced_image).item()
				else:
					niqe = 1
				if self.metric.clipiqa != 0:
					clipiqa = iqa_clipiqa(enhanced_image).item()
				else:
					clipiqa = 1
				if self.metric.maniqa != 0:
					maniqa = iqa_maniqa(enhanced_image).item()
				else:
					maniqa = 1
				if self.metric.hyperiqa != 0:
					hyperiqa = iqa_hyperiqa(enhanced_image).item()
				else:
					hyperiqa = 1
				if self.metric.musiq != 0:
					musiq = iqa_musiq(enhanced_image).item()
				else:
					musiq = 1
				if self.metric.paq2piq != 0:
					paq2piq = iqa_paq2piq(enhanced_image).item()
				else:
					paq2piq = 1
				if self.metric.brisque != 0:
					brisque = iqa_brisque(enhanced_image).item()
				else:
					brisque = 1
				if self.metric.pi != 0:
					pi = iqa_pi(enhanced_image).item()
				else:
					pi = 1
				if self.metric.nrqm != 0:
					nrqm = iqa_nrqm(enhanced_image).item()
				else:
					nrqm = 1
				if self.metric.ilniqe != 0:
					ilniqe = iqa_ilniqe(enhanced_image).item()
				else:
					ilniqe = 1
				if self.metric.entropy != 0:
					entropy = iqa_entropy(enhanced_image).item()
				else:
					entropy = 1
				if self.metric.ssim != 0:
					ssim = iqa_ssim(original_image,enhanced_image).item()
				else:
					ssim = 1
				if self.metric.psnr != 0:
					psnr = iqa_psnr(original_image,enhanced_image).item()
				else:
					psnr = 1
				if self.metric.eme != 0:
					eme = EME(enhanced_image,8,8)
				else:
					eme = 1
				if self.metric.ame != 0:
					ame = AME(enhanced_image,8,8)
				else:
					ame = 1
				fitness_measure = np.asarray([niqe, clipiqa,
											maniqa, hyperiqa,
											musiq, paq2piq,
											brisque, pi,
											nrqm, ilniqe,
											entropy, ssim, psnr,
											eme,ame])
				fitness_measure_list.append(fitness_measure)
		fitness_measure = np.asarray(fitness_measure_list)
		fitness_measure_avg = np.mean(fitness_measure,axis=0)
		total_score = 1
		for i in range(0,fitness_measure_avg.shape[0]):
			total_score = total_score * (fitness_measure_avg[i] ** self.metric_weights[i])
		return total_score,fitness_measure_avg

	def evolve(self):
		total_batch = self.data_loader.__len__(mode='train')
		for batch_count in range(0,total_batch):
			fitness_score=self.fitness_test(batch_count)
			fitness_score_total = self.aggregate_fitness_score(fitness_score)
			population_ranking = np.argsort(fitness_score_total)[::-1]
			surviving_population = []
			for i in range(0,self.parent_size):
				surviving_population.append(self.population[population_ranking[i]])
			self.population = self.next_generation(surviving_population)
		new_apex = self.population[0]
		new_apex_total_score, apex_fitness_measure_avg = self.full_dataset_test(new_apex, mode='train')
		if (self.apex != None) and (new_apex_total_score < self.apex_threshold*self.apex_score):
			self.epochs_without_improvement += 1
			self.noise_intensity = self.noise_intensity*self.noise_decay
			self.apex_timeline.append(self.apex)
			return False
		else:
			self.epochs_without_improvement = 0
			self.apex_score = new_apex_total_score
			self.apex_fitness_measure_avg = apex_fitness_measure_avg
			self.apex = new_apex
			self.apex_timeline.append(new_apex)
			return True

	def evolve_earlystop(self,max_epochs,max_epochs_without_improvement):
		for epoch in range(1,max_epochs+1):
			print("Epoch ",epoch)
			has_improved = self.evolve()
			if has_improved == True:
				print("New apex detected!")
			else:
				print("There was no improvement this epoch!")
			if max_epochs_without_improvement == self.epochs_without_improvement:
				return epoch
		return max_epochs

	def validation_score(self):
		return self.full_dataset_test(self.apex,mode='validation')

	def return_apex(self):
		return self.apex

	def return_apex_timeline(self):
		return self.apex_timeline