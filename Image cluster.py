#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torchvision.models as models
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from DataLoader import ImageDataset
from torch.utils.data import DataLoader
import pickle


# In[ ]:


csv_path = "./DL_info.csv"
Image_slices_dir = "/home/parv/Dropbox/Final_Images_2/"


# In[ ]:


vgg16 = models.vgg16(pretrained=True).features.cuda()
print (vgg16)


# In[ ]:


train_dataset = ImageDataset(root_dir = Image_slices_dir, dataset_type = 1)
#print('Dataset length: ' + str(len(new_dataset)))

dataloader = DataLoader(dataset = train_dataset, batch_size = 5)
print (len(dataloader))


# In[ ]:


curr_features = []
filenames = []
batch_no = 0
for batch in dataloader:
    out = list(vgg16(batch['image'].cuda()).view(-1, 512*16*16).cpu().detach().numpy())
    # print (out)
    curr_features.extend(out)
    # print (batch['Filename'])
    filenames.extend(batch['Filename'])
    batch_no+=1
    print (batch_no)
#     if batch_no==10:
#         break


# In[ ]:


cluster_function = KMeans(n_clusters= 8)
assigned_clusters = cluster_function.fit_predict(curr_features)


# In[ ]:


#pickle.dump(cluster_function,open('./cluster_function.p','wb'))
#pickle.dump(assigned_clusters, open('./assigned_clusters.p','wb'))
pickle.dump(filenames, open('./filenames.p','wb'))
pickle.dump(curr_features, open('./curr_features', 'wb'))


# In[ ]:


assigned_clusters


# In[ ]:


len (filenames)


# In[ ]:


assigned_clusters_2 = pickle.load(open('./assigned_clusters.p','rb'))


# In[ ]:


assigned_clusters_2


# In[ ]:




