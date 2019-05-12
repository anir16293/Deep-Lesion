#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
import sys
from chainercv.datasets import VOCBboxDataset
from chainercv.datasets import voc_bbox_label_names
from chainercv.visualizations import vis_bbox
from chainer import iterators
import torch
import torch.nn as nn
import torch.nn.functional as functional
from DataLoader import ImageDataset
import numpy as np
from panet import PanNet
from panet import IOULoss

from panet import EncoderNet


# In[2]:


csv_path = "./DL_info.csv"
Image_slices_dir = "/home/parv/Dropbox/Final_Images_2/"
filename = "./Models/ValidationLoss.csv"


# In[3]:


train_dataset = ImageDataset(root_dir = Image_slices_dir, dataset_type=1)
validation_dataset = ImageDataset(root_dir = Image_slices_dir, dataset_type=2)
test_dataset = ImageDataset(root_dir = Image_slices_dir, dataset_type=3)


# In[4]:


#train_dataset[0]


# In[4]:


##using dataloader for batch trainig 
batch_size=2
dataloader_train=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
print (len(dataloader_train))

dataloader_validation=torch.utils.data.DataLoader(validation_dataset,batch_size=batch_size,shuffle=False)
print (len(dataloader_validation))

dataloader_test=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
print (len(dataloader_test))


# In[6]:


# Just a debug loop
# for i in range(100):
#     train_dataset[i]


# In[5]:


# Load the model
model = PanNet().cuda()
# model.load_state_dict(torch.load('./model_'))
# model.cuda()


# In[ ]:





# In[6]:


# The parameters for learning
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
criterion2 = nn.MSELoss()
criterion1 = IOULoss()

num_epochs = 5


# In[7]:


Running_training_loss_list = []
Validation_loss_list = []

def validation_loop():
    loss_validation = 0
    count = 0
    for sample_batched in dataloader_validation:
        model.eval()
        # out1, out2 = model(sample_batched['image'].cuda())
        out1 = model(sample_batched['image'].cuda())
        loss = criterion2(out1, sample_batched['lesions'].cuda()) #+ (1e-6)*criterion2(out2, sample_batched['image'].cuda())
        count+=1
        Loss_val = loss.cpu().detach().item()
        loss_validation = loss_validation + Loss_val
        torch.cuda.empty_cache()
#         if count==10:
#             break
    final_loss = (loss_validation*1.0)/count
    print("===============================")
    print("Validation loss is ", final_loss)
    print("===============================")
    Validation_loss_list.append(final_loss)


# In[8]:


############################################################
######Training Loop#########################################
for epoch in range(num_epochs):
    Running_loss = 0
    count = 0
    
    for sample_batched in dataloader_train:
        model.train()
#         print (sample_batched['image'].dtype)
#         print (sample_batched['lesions'].dtype)
#         out1, out2 = model(sample_batched['image'].cuda())
        out1 = model(sample_batched['image'].cuda())
        print (out1)
        print(sample_batched['lesions'])
        #print (out.shape)
        #print (sample_batched['lesions'].shape)
        loss = criterion2(out1, sample_batched['lesions'].cuda()) #+ (1e-6)*criterion2(out2, sample_batched['image'].cuda())
        # Store val in outer variable for printing
#         print (out)
        Loss_val = loss.cpu().detach().item()
        Running_loss = Running_loss + Loss_val
#         print (out[0])
#         print (sample_batched['lesions'][0])
        count += 1
        print ("Batch Loss", Loss_val)
        print ("Running Loss", Running_loss/count)
        #### Measure per image running loss for training set EVERY 50 batches
#         if count%50==0:
#             Running_training_loss_list.append(Running_loss/(count))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
#         if count==10:
#             break
        #### Measure per image running loss for validation set EVERY 50 batches
#         if count%50 == 0:
#             sample_batched['image'].cpu().detach()
#             sample_batched['lesions'].cpu().detach()
#             validation_loop()
#         scheduler.step()

#     data0.cpu().detach()
#     data1.cpu().detach()
#     loss.cpu().detach()
    
    # Save model to disk for the epoch
    model.cpu()
    torch.save(model.state_dict(), './Models/panet_model_'+str(epoch))
    model.cuda()
    scheduler.step()
    
    # Call validation loop
    validation_loop()
    
    ### open and write to csv
    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, Validation_loss_list[epoch]])
    f.close()
    Running_training_loss_list.append(Running_loss/count)
    print ("Running Loss", Running_loss/count)


# In[12]:


model.cpu()
torch.save(model.state_dict(), './model_')


# In[ ]:





# In[17]:


# model = model.cpu()
# count = 0
# loss_validation = 0
# for sample_batched in dataloader_test:
#     model.eval()
#     out = model(sample_batched['image'])
#     print (out[0].shape)
#     print (sample_batched['lesions'].shape)

#     loss = criterion1(out[0], sample_batched['lesions'])
#     count+=1
#     Loss_val = loss.cpu().detach().item()
#     print ("Batch Loss", Loss_val)
#     loss_validation = loss_validation + Loss_val
#     print ("Running Loss", loss_validation/count)
    


# In[20]:


# #model = model.cpu()
# for sample_batched in dataloader:
#     img = sample_batched['image'][0]
#     print (img.shape)
# #    img = img.view(512, 512, 3)
# #    img = img.numpy().transpose(1,2,0)
#     plt.imshow(img)
#     plt.show()


# In[ ]:




