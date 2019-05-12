import os
import shutil

root_dir='Data/';
copy_dir='/home/weilunhuang/faster-rcnn-pytorch/VOC2007/ImageSets/Main';

for i in range(1,9):
    sub_path='class'+str(i);
    os.makedirs(root_dir+sub_path+'/JPEGImages');
    os.makedirs(root_dir+sub_path+'/Annotations');
    os.makedirs(root_dir+sub_path+'/ImageSets/Main');
    dest_dir=root_dir+sub_path+'/ImageSets/Main';
    file_name=copy_dir+'/test.txt';
    shutil.copy(file_name, dest_dir);

    file_name=copy_dir+'/train.txt';
    shutil.copy(file_name, dest_dir);

    file_name=copy_dir+'/val.txt';
    shutil.copy(file_name, dest_dir);
