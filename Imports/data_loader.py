import numpy as np
import tensorflow as tf
import PIL
import math
import os
from inference import image_haze_removal
import cv2
from enhancement_algorithms import UM,HEF,CLAHE,ATACE,TCDHE,TCDHE

class CustomDataLoader(tf.keras.utils.Sequence):

  def __init__(self, classifier_data_loader, batch_size, image_shape, acceptable_labels, mode='train'):
    self.classifier_data_loader = classifier_data_loader
    self.batch_size = batch_size
    self.mode = mode
    self.image_shape = image_shape
    self.acceptable_labels = acceptable_labels

  def __len__(self):
    #return math.ceil(len(self.x) / self.batch_size)
    return (self.classifier_data_loader.len(mode=self.mode) // self.batch_size)

  def __getitem__(self, idx):
    low = idx * self.batch_size
    # Cap upper bound at array length; the last batch may be smaller
    # if the total number of items is not a multiple of batch size.
    high = min(low + self.batch_size, self.classifier_data_loader.len(mode=self.mode))
    batch_x = []
    batch_y = []

    batch_image,batch_label = self.classifier_data_loader.get_batch(low,high-low,mode=self.mode)

    for i in range(low,high):
      image = batch_image[i]
      label = batch_label[i]
      image_pil = PIL.Image.fromarray(image)
      image_pil = image_pil.resize(self.image_shape,PIL.Image.Resampling.BILINEAR)
      image_pil = image_pil.convert('RGB')
      batch_x.append(np.asarray(image_pil))
      for j in range(0,len(self.acceptable_labels)):
        if self.acceptable_labels[j] == label:
          batch_y.append(j)

    batch_x = np.asarray(batch_x)
    batch_y = np.asarray(tf.keras.utils.to_categorical(batch_y,len(self.acceptable_labels)))

    return batch_x,batch_y

class folder_data_loader():

  def __init__(self, class_path_list, img_size, batch_size, train_ratio = 1.0, dataset_size_scaling = 1.0, ld_net=None):

    self.batch_size = batch_size
    self.x = []
    self.y = []
    self.class_representation = np.zeros(len(class_path_list))
    
    for label,path in enumerate(class_path_list):
      for filename in os.listdir(path):
        image = PIL.Image.open(os.path.join(path,filename))
        if image is not None:
          image = PIL.ImageOps.grayscale(image)
          image = image.resize(img_size,PIL.Image.Resampling.BILINEAR)
          image = np.asarray(image)
          if ld_net != None:
            if ld_net == 'clahe':
              image = CLAHE(image,2.0,8)
            elif ld_net == 'um':
              image = UM(image,5,2)
            elif ld_net == 'hef':
              image = HEF(image,20)
            elif ld_net == 'atace':
              image = ATACE(image)
            elif ld_net == 'tcdhe':
              try:
                image = TCDHE(image)
              except:
                image = image
            else:
              image = image_haze_removal((image/255.0).astype(np.single),ld_net)
              image = np.asarray(image)
          self.x.append(image)
          self.y.append(label)
          self.class_representation[label] = self.class_representation[label]+1
    self.x = np.asarray(self.x)
    self.y = np.asarray(self.y)
    self.dataset_size = self.y.shape[0]
    self.train_size = int(self.dataset_size*train_ratio)
    for class_id in range(0,len(class_path_list)):
      class_index = np.where(self.y == class_id)[0]
      np.random.shuffle(class_index)
      class_index = class_index[:int(class_index.shape[0]*dataset_size_scaling)]
      if class_id == 0:
        self.train_index_list = class_index[0:int(class_index.shape[0]*train_ratio)]
        self.validation_index_list = class_index[int(class_index.shape[0]*train_ratio):]
      else:
        self.train_index_list = np.append(self.train_index_list, class_index[0:int(class_index.shape[0]*train_ratio)])
        self.validation_index_list = np.append(self.validation_index_list, class_index[int(class_index.shape[0]*train_ratio):])
    #index_list = np.arange(0,self.dataset_size,1,int)
    #np.random.shuffle(index_list)
    #self.train_index_list = index_list[0:int(self.train_size*dataset_size_scaling)]
    #self.validation_index_list = index_list[int(self.train_size*dataset_size_scaling):int(self.dataset_size*dataset_size_scaling)]
  def image_enhancement(self,image_enhancement_method):
    if image_enhancement_method == 'clahe':
      clipLimit = 2.0
      raio = 8
      clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(raio,raio))
      return clahe.apply(image)



  def __len__(self,mode='train'):
    if mode=='train':
      return math.ceil(self.train_index_list.shape[0] / self.batch_size)
    else:
      return math.ceil(self.validation_index_list.shape[0] / self.batch_size)

  def get_item(self, index, mode='train'):
    low = index * self.batch_size
    high = min(low + self.batch_size, self.train_index_list.shape[0])

    if mode=='train':
      real_index = self.train_index_list[low:high]
    else:
      real_index = self.validation_index_list[low:high]
    batch_x = (self.x[real_index]/255.0).astype(np.single)
    batch_y = self.y[real_index]
    batch_y = np.asarray(tf.keras.utils.to_categorical(batch_y,self.class_representation.shape[0]))

    return batch_x,batch_y