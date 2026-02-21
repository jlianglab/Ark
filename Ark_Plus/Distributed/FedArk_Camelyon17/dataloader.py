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
import pandas as pd
import openpyxl
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, HueSaturationValue,
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
      transformations_list.append(transforms.RandomVerticalFlip())
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
      ShiftScaleRotate(rotate_limit=180),
      HorizontalFlip(p=0.5),
      VerticalFlip(p=0.5),
      OneOf([
          RandomBrightnessContrast(),
          RandomGamma(),
           ], p=0.3),
    ])
    return AUGMENTATIONS


class Camelyon17(Dataset):

  def __init__(self, data_path, split, source, crop_size=96, resize=112, augment=None, few_shot = -1):

    self.data_path = data_path

    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)
    
    assert split in ['train', 'test']
    assert source in ['hospital1', 'hospital2', 'hospital3', 'hospital4', 'hospital5'] # five hospital

    data_dict = np.load(os.path.join(data_path, 'data.pkl'), allow_pickle=True)
    self.paths, self.labels = data_dict[source][split]

    self.labels = self.labels.astype(np.long).squeeze()


    # indexes = np.arange(self.paths.shape[0])
    # if few_shot > 0:
    #     random.Random(99).shuffle(indexes)
    #     num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
    #     indexes = indexes[:num_data]
    #     _img_list= copy.deepcopy(self.img_list)
    #     _img_label= copy.deepcopy(self.img_label)
    #     self.img_list = []
    #     self.img_label = []
    #     for i in indexes:
    #         self.img_list.append(_img_list[i])
    #         self.img_label.append(_img_label[i])
    #     print(f"{few_shot} of total: {len(self.img_list)}")
        
    
  def __getitem__(self, index):
    cv2.setNumThreads(0)
    img_path = os.path.join(self.data_path, self.paths[index])
    imageData = Image.open(img_path).convert('RGB')
    
    imageLabel = [0.0, 0.0]   
    imageLabel[self.labels[index]] = 1.0
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
    
    return student_img, teacher_img, imageLabel

  def __len__(self):
      return self.paths.shape[0]

dict_dataloarder = {
    "Camelyon17": Camelyon17
}
