import os
import torch
import random
import copy
import csv
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import pydicom as dicom
import cv2
from skimage import transform, io, img_as_float, exposure
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop, RandomResizedCrop, Normalize
)
from albumentations.pytorch import ToTensorV2

def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def build_ts_transformations(crop_size):
    AUGMENTATIONS = Compose([
      RandomResizedCrop(height=crop_size, width=crop_size),
      ShiftScaleRotate(rotate_limit=10),
      OneOf([
          RandomBrightnessContrast(),
          RandomGamma(),
           ], p=0.3),
    ])
    return AUGMENTATIONS


class ChestXray14(Dataset):

  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=14, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)
    

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((self.resize,self.resize))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


# ---------------------------------------------Downstream CheXpert------------------------------------------
class CheXpert(Dataset):

  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)
    
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        if "test" in line[0]:
          label = line[1:]
        else:
          label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              if self.uncertain_label == "Ones":
                label[i] = 1
              elif self.uncertain_label == "Zeros":
                label[i] = 0
              elif self.uncertain_label == "LSR-Ones":
                label[i] = random.uniform(0.55, 0.85)
              elif self.uncertain_label == "LSR-Zeros":
                label[i] = random.uniform(0, 0.3)
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((self.resize,self.resize))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream Shenzhen------------------------------------------
class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=1, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((self.resize,self.resize))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream VinDrCXR------------------------------------------
class VinDrCXR(Dataset):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)

    with open(file_path, "r") as fr:
      line = fr.readline().strip()
      while line:
        lineItems = line.split()
        imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
        imageLabel = lineItems[1:]
        imageLabel = [int(i) for i in imageLabel]
        self.img_list.append(imagePath)
        self.img_label.append(imageLabel)
        line = fr.readline()

    if annotation_percent < 100:
      indexes = np.arange(len(self.img_list))
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData.resize((self.resize,self.resize)))).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel
    
  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream RSNA Pneumonia------------------------------------------
class RSNAPneumonia(Dataset):

  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.strip().split(' ')
          imagePath = os.path.join(images_path, lineItems[0])


          self.img_list.append(imagePath)
          imageLabel = [0, 0, 0]
          imageLabel[int(lineItems[-1])] = 1
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((self.resize,self.resize))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)

dict_dataloarder = {
    "ChestXray14": ChestXray14,
    "CheXpert": CheXpert,
    "Shenzhen": ShenzhenCXR,
    "VinDrCXR": VinDrCXR,
    "RSNAPneumonia": RSNAPneumonia
}


class OmniPretrainingDatasets(Dataset):
  def __init__(self, datasets_config, dataset_list = ["ChestXray14"], crop_size=224, resize=256, augment=None):
    self.dataset_list = dataset_list
    self.datasets_config = datasets_config
    self.dataset_image_list = []
    self.dataset_label_list = []
    self.dataset_index_list = []

    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)
    
    self.num_classes_list = []
    for idx, dataset in enumerate(dataset_list):
        dataset_loaded = dict_dataloarder[dataset](images_path=self.datasets_config[dataset]['data_dir'], file_path=self.datasets_config[dataset]['train_list'], augment=None)
        self.dataset_image_list.extend(dataset_loaded.img_list)
        self.dataset_label_list.extend(dataset_loaded.img_label)
        self.dataset_index_list.extend([idx for _ in range(len(dataset_loaded.img_list))])
        self.num_classes_list.append(len(self.datasets_config[dataset]['diseases']))
  
    max_class_num = max(self.num_classes_list)
    print("max_class_num", max_class_num)
    label_padding = []
    for label in self.dataset_label_list:
        if len(label) < max_class_num:
          label.extend([0 for _ in range(max_class_num - len(label))])
          assert len(label) == max_class_num
        label_padding.append(label)


  def __getitem__(self, index):
    cv2.setNumThreads(0)

    image_path = self.dataset_image_list[index]
    imageData = Image.open(image_path).convert('RGB').resize((self.resize,self.resize))
    imageLabel = self.dataset_label_list[index]
    imageLabel = torch.FloatTensor(imageLabel)
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.     
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
  
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')

    return student_img, teacher_img, imageLabel, self.dataset_index_list[index]

  def __len__(self):
    return len(self.dataset_image_list)


class OmniPretrainingDatasets_EqualSampling(Dataset):
  def __init__(self, datasets_config, dataset_list = ["ChestXray14"], normalization = "imagenet"):
    self.dataset_list = dataset_list
    self.datasets_config = datasets_config
    self.dataset_image_lists = []
    self.dataset_label_lists = []
    self.num_classes_list = []
    for dataset in dataset_list:
        dataset_loaded = dict_dataloarder[dataset](images_path=self.datasets_config[dataset]['data_dir'], file_path=self.datasets_config[dataset]['train_list'], augment=None)
        self.dataset_image_lists.append(dataset_loaded.img_list)
        self.dataset_label_lists.append(dataset_loaded.img_label)
        self.num_classes_list.append(len(self.datasets_config[dataset]['diseases']))
    self.data_number_list = [len(im_list) for im_list in self.dataset_image_lists]
    self.prime_length = max(self.data_number_list) # set the dataset with the most number of data to be prime
    self.augment = build_transform_classification(normalize = normalization, mode="train")

  def __getitem__(self, index):
    # dataset_list ["ChestXray14", "CheXpert", "VinDrCXR"] 
    # data_number_list [70K, 220K, 15K]
    image_data_list = []
    image_label_list = []
    # return one image from each dataset to assemle a batch
    for i in range(len(self.dataset_list)):
      # this is the prime dataset with the most number of data
      if self.data_number_list[i] == self.prime_length: 
        reindex = index
      #  when self.data_number_list[i] < self.prime_length, need to deal with the index
      elif self.data_number_list[i] < self.prime_length: 
        reindex = index % self.data_number_list[i]
        if reindex == 0:
          random.Random(0).shuffle(self.dataset_image_lists[i])
          random.Random(0).shuffle(self.dataset_label_lists[i])
          
      image_path = self.dataset_image_lists[i][reindex]
      image_data = Image.open(image_path).convert('RGB')
      image_label = torch.FloatTensor(self.dataset_label_lists[i][reindex])
      if self.augment != None: image_data = self.augment(image_data)
      image_data_list.append(image_data)
      image_label_list.append(image_label)

    return image_data_list, image_label_list

  def __len__(self):
    return self.prime_length