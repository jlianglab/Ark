import os
import torch
import random
import copy
import csv
from PIL import Image
import json
import SimpleITK as sitk

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
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)



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


class ChestXray14(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment

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
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)


# ---------------------------------------------Downstream CheXpert------------------------------------------
class CheXpert(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment
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
              label[i] = -1
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        imageLabel = [int(i) for i in label]
        self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    label = []
    for l in self.img_label[index]:
      if l == -1:
        if self.uncertain_label == "Ones":
          label.append(1)
        elif self.uncertain_label == "Zeros":
          label.append(0)
        elif self.uncertain_label == "LSR-Ones":
          label.append(random.uniform(0.55, 0.85))
        elif self.uncertain_label == "LSR-Zeros":
          label.append(random.uniform(0, 0.3))
      else:
        label.append(l)
    imageLabel = torch.FloatTensor(label)

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream Shenzhen------------------------------------------
class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=1, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment

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
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream VinDrCXR------------------------------------------
class VinDrCXR(Dataset):
    def __init__(self, images_path, file_path, augment, num_class=6, few_shot = -1):
        self.img_list = []
        self.img_label = []
        self.augment = augment

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

        indexes = np.arange(len(self.img_list))
        if few_shot > 0:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _img_list= copy.deepcopy(self.img_list)
            _img_label= copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []
            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])
            print(f"{few_shot} of total: {len(self.img_list)}")

    def __getitem__(self, index):

        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel
    def __len__(self):
        return len(self.img_list)
    
class VinDrCXR_all(Dataset):
    def __init__(self, images_path, file_path, diseases, augment = None, few_shot = -1):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            if "train" in file_path:
                all_diseases = next(csvReader, None)[2:]
                disease_idxs = [all_diseases.index(d) for d in diseases]
                # print(diseases)
                # print(disease_idxs)
                lines = [line for line in csvReader]
                assert len(lines)/3 == 15000
                for i in range(15000):
                    imagePath = os.path.join(images_path, "train_jpeg", lines[i*3][0]+".jpeg")
                    label = [0 for _ in range(len(diseases))]
                    r1,r2,r3 = lines[i*3][2:],lines[i*3+1][2:],lines[i*3+2][2:] 
                    for c in disease_idxs:
                        label[c] = 1  if int(r1[c])+int(r2[c])+int(r3[c]) > 0 else 0
                    self.img_list.append(imagePath)
                    self.img_label.append(label)
            else:
                all_diseases = next(csvReader, None)[1:]
                disease_idxs = [all_diseases.index(d) for d in diseases]
                # print(diseases)
                # print(disease_idxs)
                for line in csvReader:
                    imagePath = os.path.join(images_path, "test_jpeg", line[0]+".jpeg")
                    label = [int(l) for l in line[1:]]
                    # label = label[disease_idxs]
                    self.img_list.append(imagePath)
                    self.img_label.append(label)
        
        print("label shape: ", np.array(self.img_label).shape, np.sum(np.array(self.img_label), axis=0))

        indexes = np.arange(len(self.img_list))
        if few_shot > 0:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
            indexes = indexes[:num_data]
            _img_list= copy.deepcopy(self.img_list)
            _img_label= copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []
            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])
            print(f"{few_shot} of total: {len(self.img_list)}")

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        imageData = Image.open(imagePath).convert('RGB')
        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel
    def __len__(self):
        return len(self.img_list)


# ---------------------------------------------Downstream RSNA Pneumonia------------------------------------------
class RSNAPneumonia(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=3, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.strip().split(' ')
          imagePath = os.path.join(images_path, lineItems[0])


          self.img_list.append(imagePath)
          self.img_label.append(int(lineItems[-1]))

    indexes = np.arange(len(self.img_list))
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = np.zeros(3)
    imageLabel[self.img_label[index]] = 1
    imageLabel = torch.FloatTensor(imageLabel)
    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream COVIDx------------------------------------------
class COVIDx(Dataset):

  def __init__(self, images_path, file_path, augment, classes, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          patient_id, fname, label, source  = line.strip().split(' ')
          imagePath = os.path.join(images_path, fname)

          self.img_list.append(imagePath)
          self.img_label.append(classes.index(label))

    indexes = np.arange(len(self.img_list))
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = np.zeros(3)
    imageLabel[self.img_label[index]] = 1
    imageLabel = torch.FloatTensor(imageLabel)
    if self.augment != None: imageData = self.augment(imageData)
 
    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream MIMIC------------------------------------------
class MIMIC(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
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
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)

class ChestDR(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=19, few_shot = -1):

    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0]+'.png')
          imageLabel = lineItems[1].split(',')
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    
    if few_shot > 0:
        random.Random(99).shuffle(indexes)
        num_data = int(indexes.shape[0] * few_shot) if few_shot < 1 else int(few_shot)
        indexes = indexes[:num_data]
        _img_list= copy.deepcopy(self.img_list)
        _img_label= copy.deepcopy(self.img_label)
        self.img_list = []
        self.img_label = []
        for i in indexes:
            self.img_list.append(_img_list[i])
            self.img_label.append(_img_label[i])
        print(f"{few_shot} of total: {len(self.img_list)}")


  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)
