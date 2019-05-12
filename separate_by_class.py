##########
This script is used to separate out the dataset into different folders by classes
##########
import os

import numpy as np
import pandas as pd
import re
import shutil

##### For all paths
csv_path = "./DL_info.csv"
Image_slices_dir = "/home/weilunhuang/course/DL/project/faster-rcnn-pytorch/data/JPEGImages"
Anno_slices_dir = "/home/weilunhuang/course/DL/project/faster-rcnn-pytorch/data/Annotations"
ImageSet_slices_dir = "/home/weilunhuang/course/DL/project/faster-rcnn-pytorch/data/ImageSets/Main"

root_dir="/home/weilunhuang/course/DL/project/Data"

###create dict for 8 classes
dest_dir={};
for i in range(1,9):
    sub_path="/class"+str(i);
    dest_dir[i]={};
    #3 folders for JPEG, Annotations, ImageSets 
    dest_dir[i]['jpeg']=root_dir+sub_path+'/JPEGImages';
    dest_dir[i]['annotation']=root_dir+sub_path+'/Annotations';
    dest_dir[i]['imagesets']=root_dir+sub_path+'/ImageSets/Main';


###df stuff
df = pd.read_csv(csv_path)          
test_df = df[df['Train_Val_Test']==3];
df_dic={};

for i in range(1,9):
    df_dic[i]=test_df[df['Coarse_lesion_type']==i];

###separate images and annotations
for i in range(1,9):
    for j in range(len(df_dic[i].index)):
        #debug
        if j>0:
            break;
        #jpeg
        file_name = os.path.join(Image_slices_dir, df_dic[i].iloc[j]['File_name'])        
        print (file_name)
        shutil.copy(file_name, dest_dir[i]['jpeg'])
        
        #xml
        file_name = os.path.join(Anno_slices_dir, df_dic[i].iloc[j]['File_name'])
        file_name,_=os.path.splitext(file_name)
        file_name=file_name+'.xml'
        print(file_name);
        shutil.copy(file_name, dest_dir[i]['annotation'])

###separate imagesets, for test dataset
for i in range(1,9):
    #debug
    if i>1:
        break;
    path=dest_dir[i]['imagesets']+'/test.txt'
    file = open(path,"w") ;
    for j in range(len(df_dic[i].index)):
        idx=df_dic[i].iloc[j]['File_name'];
        idx,_=os.path.splitext(idx)
        file.write(idx+'\n');
    file.close();
