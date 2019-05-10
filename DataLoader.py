#!/usr/bin/env python
# coding: utf-8

# In[1]:


#############
# This notebook represents the custom dataloader to read the DeepLesion Dataset
#############

import torchvision.models as models

import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch.utils.data.dataset as dataset
import cv2
import re
import matplotlib.patches as patches
import pandas as pd

# In[2]:


##### For all paths
csv_path = "./DL_info.csv"
Image_slices_dir = "/home/parv/Dropbox/Final_Images_2/"


# In[7]:


#### Optional, remove it later, put the same into dataset class
df = pd.read_csv(csv_path)          # The DL_info.csv file path
df.sort_values("File_name", inplace=True) 
df.drop_duplicates(subset ="File_name", 
                     keep = False, inplace = True) 
print (len(df.index))
# print(df)
train_df = df[df['Train_Val_Test']==1]
validation_df = df[df['Train_Val_Test']==2]
test_df = df[df['Train_Val_Test']==3]

print(len(train_df.index))
print(len(validation_df.index))
print(len(test_df.index))

print(len(train_df.index)+len(validation_df.index)+len(test_df.index))


# In[3]:


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        image, lesions = sample['image'], sample['lesions']
#         plt.figure()
#         plt.imshow(image)
#         print("Image dtype is ", image.dtype)
        image = image.astype(np.float32)
#         for i in range(lesions.shape[0]):
#             plt.gca().add_patch(plt.Rectangle((lesions[i][0],lesions[i][1]),
#                                 lesions[i][2]-lesions[i][0],
#                                 lesions[i][3]-lesions[i][1],
#                                 linewidth=1,edgecolor='r', fill=False))
#         # plt.scatter(lesions[:, 0], lesions[:, 1], s=10, marker='.', c='r')
#         plt.show()
#         print(image.shape)
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            elif h<w:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        # print ("New hieght is ", new_h, new_w)
        img = cv2.resize(image, (new_h, new_w))
        
        assert new_h==new_w, "Width Size not same"
        # h and w are swapped for lesions because for images,
        # x and y axes are axis 1 and 0 respectively
#         print("Image dtype now is ", img.dtype)
        lesions = lesions * [new_w / w, new_h / h, new_w / w, new_h / h]
#         plt.figure()
# #         plt.cla()
#         plt.imshow(img)
#         for i in range(lesions.shape[0]):
#             plt.gca().add_patch(plt.Rectangle((lesions[i][0],lesions[i][1]),
#                                 lesions[i][2]-lesions[i][0],
#                                 lesions[i][3]-lesions[i][1],
#                                 linewidth=1,edgecolor='r', fill=False))
#         plt.show()
#         print (img.shape)
#         print ("lesions are", lesions)
        lesions = lesions.astype(np.float32)
        return {'image': img, 'lesions': lesions, 'labels':np.ones(lesions.shape[0])}


# In[4]:


class ToTensor(object):
    def __call__(self, sample):
        image, lesions = sample['image'], sample['lesions']
        # print ("Image type is", type(image))
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'lesions': torch.from_numpy(lesions),
                'labels':np.ones(lesions.shape[0])}


# In[5]:


img_transform = transforms.Compose([
    Rescale(512),
    ToTensor(),
])


# dataset_type : 1 for train, 2 for validation, 3 for test
class ImageDataset(dataset.Dataset):
    def __init__ (self, root_dir, dataset_type, csv_file=csv_path, transform=img_transform):
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.df = df[df['Train_Val_Test']==dataset_type]
        self.imf_transform = img_transform
    def __len__(self):
        return len(self.df.index)
    def __getitem__(self, idx):
        # read file
        file_name = os.path.join(self.root_dir, self.df.iloc[idx]['File_name'])        
        image = cv2.imread(file_name)
        
        # print(image.shape)
#        print (idx)
#        assert image.shape==(512, 512, 3), "Input size does not match"
        # show image
#         plt.figure()
#         plt.imshow(image)
#         plt.show()
        # print(self.df.iloc[idx]['File_name'])
        #find all boudning boxes
        lesions = []
        new_df = self.df[self.df['File_name']==self.df.iloc[idx]['File_name']]
        for i in range(len(new_df.index)):
            coordinates_str = (re.split(',',new_df.iloc[i]['Bounding_boxes']))
            coordinates = [ float(x) for x in coordinates_str]
            # print (coordinates)
            lesions.append(coordinates)
        lesions = np.asarray(lesions)
        #print (lesions)
        #print (lesions.shape)
        sample = {'image':image, 'lesions':lesions, 'labels':np.ones(lesions.shape[0])}
        # print (type(lesions))
        sample = img_transform(sample)
        return sample


# In[8]:


train_dataset = ImageDataset(root_dir = Image_slices_dir, dataset_type=1)
validation_dataset = ImageDataset(root_dir = Image_slices_dir, dataset_type=2)
test_dataset = ImageDataset(root_dir = Image_slices_dir, dataset_type=3)

# print(len(train_dataset))
# print(len(validation_dataset))
# print(len(test_dataset))


# In[12]:


# example = train_dataset[28]
# print (type(example["image"]))
# print (type(example["lesions"]))
# print (example["labels"].shape)
# train_dataset[54]
# for i in range(0, len(train_dataset)):
#     train_dataset[i]
    


# In[ ]:


# for i in range(0, len(validation_dataset)):
#     validation_dataset[i]


# In[ ]:


# for i in range(0, len(test_dataset)):
#     test_dataset[i]

