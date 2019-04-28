#!/usr/bin/env python
"""
This code has been adapted from the original code authored by:
Ke Yan
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
May 2018
"""

"""
This code is a part of the DeepLesion course project for the Deep Learning course (CS 682), Spring 2019 at John's Hopkins University.
The project team consists of Aniruddha Tamhane, Parv Saxena and Wei-Lun Huang.
The course was taught by Matthias Unberath and TA'd by Jie-Ying Wu and Gao Cong.
The project was sponsored by Google Cloud and Intuitive Surgicals.
"""


import numpy as np
import nibabel as nib
import os
import cv2
import csv
import sys
import pandas as pd
#import matplotlib.pyplot as plt
#import shutil


#dir_in = '/Users/aniruddha/Downloads/CT_files_download/Images_png' #Same as unzip_directory
#dir_out = '/Users/aniruddha/Downloads/CT_files_nifti'              # Directory that will store the 3-channel Nifti images
out_fmt = '%s_%03d-%03d.nii.gz'                                    # format of the nifti file name to output. Do not change it.
#info_fn = '/Users/aniruddha/Downloads/DL_info.csv'                 # file name of the information file

"""
Command:  python3 Deep-Lesion/DL_save_nifti.py /media/parv/Seagate\ Backup\ Plus\ Drive/DeepL_Dataset/Extracted\ /55/Images_png/ /media/parv/Seagate\ Backup\ Plus\ Drive/DeepL_Dataset/Final_nifti/ Deep-Lesion/DL_info.csv 
"""
dir_in = str(sys.argv[1])
dir_out = str(sys.argv[2])
info_fn = str(sys.argv[3])

info_file = pd.read_csv(info_fn)
key_slice_set = set(info_file.File_name.values)
lesion_mapper = {'bone':1, 'abdomen':2, 'mediastinum': 3,'liver':4, 'lung':5, 'kidney': 6, 'soft_tissue':7, 'pelvis':8}
relu = lambda x: x if x>0 else 0
relu_vector = np.vectorize(relu)

def slices2nifti(ims, fn_out, spacing):
    """save 2D slices to 3D nifti file considering the spacing"""
    if len(ims) < 300:  # cv2.merge does not support too many channels
        V = cv2.merge(ims)
    else:
        V = np.empty((ims[0].shape[0], ims[0].shape[1], len(ims)))
        for i in range(len(ims)):
            V[:, :, i] = ims[i]

    # the transformation matrix suitable for 3D slicer and ITK-SNAP
    T = np.array([[0, -spacing[1], 0, 0], [-spacing[0], 0, 0, 0], [0, 0, -spacing[2], 0], [0, 0, 0, 1]])
    img = nib.Nifti1Image(V, T)
    print(img.shape)
    path_out = os.path.join(dir_out, fn_out)
    nib.save(img, path_out)
    print (fn_out, 'saved')


def load_slices(dir, slice_idxs):
    """load slices from 16-bit png files"""
    slice_idxs = np.array(slice_idxs)
    assert np.all(slice_idxs[1:] - slice_idxs[:-1] == 1)
    ims = []
    counter = 0
    slice_counter = 0
    #key_slice = int((len(slice_idxs)-1)/2)
    for slice_idx in slice_idxs:
        fn = '%03d.png' % slice_idx
        path = os.path.join(dir_in, dir, fn)
        dir_path = os.path.join(dir_in, dir)
        img_name = '_'.join([dir, fn])
        
        if img_name in key_slice_set:
            im = cv2.imread(path, -1)  # -1 is needed for 16-bit image
            assert im is not None, 'error reading %s' % path
            print ('read', path)
            im = (im.astype(np.int32) - 32768).astype(np.int16)
            im = ((im + 1024)/(1024 + 3071))*255
            im = relu_vector(im)
            im = im.astype(np.int8)
            #ims.append((im.astype(np.int32) - 32768).astype(np.int16))
            ims.append(im)
            slice_counter = counter
            counter += 1
        else:
            counter += 1
    for slice_idx in [slice_idxs[slice_counter - 1],
                      slice_idxs[slice_counter + 1]]:
        fn = '%03d.png' % slice_idx
        path = os.path.join(dir_in, dir, fn)
        dir_path = os.path.join(dir_in, dir)
        img_name = '_'.join([dir, fn])
        counter = 0
        slice_counter = 0
        im = cv2.imread(path, -1)  # -1 is needed for 16-bit image
        assert im is not None, 'error reading %s' % path
        print('read', path)
        im = (im.astype(np.int32) - 32768).astype(np.int16)
        im = ((im + 1024)/(1024 + 3071))*255
        im = relu_vector(im)
        im = im.astype(np.int8)
        #ims.append((im.astype(np.int32) - 32768).astype(np.int16))
        ims.append(im)

    return (ims)


def read_DL_info():
    """read spacings and image indices in DeepLesion"""
    spacings = []
    idxs = []
    with open(info_fn, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        rownum = 0
        for row in reader:
            if rownum == 0:
                header = row
                rownum += 1
            else:
                idxs.append([int(d) for d in row[1:4]])
                spacings.append([float(d) for d in row[12].split(',')])

    idxs = np.array(idxs)
    spacings = np.array(spacings)
    return idxs, spacings


if __name__ == '__main__':

    idxs, spacings = read_DL_info()
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    img_dirs = os.listdir(dir_in)
    img_dirs.sort()
    
    for dir1 in img_dirs:
        # find the image info according to the folder's name
        if dir1 != '.DS_Store':
            idxs1 = np.array([int(d) for d in dir1.split('_')])
            i1 = np.where(np.all(idxs == idxs1, axis=1))[0]
            spacings1 = spacings[i1[0]]

            folder_path = os.path.join(dir_in, dir1)
            fns = os.listdir(os.path.join(dir_in, dir1))
            slices = [int(d[:-4]) for d in fns if d.endswith('.png')]
            slices.sort()

            # Each folder contains png slices from one series (volume)
            # There may be several sub-volumes in each volume depending on the key slices
            # We group the slices into sub-volumes according to continuity of the slice indices
            groups = []
            for slice_idx in slices:
                if len(groups) != 0 and slice_idx == groups[-1][-1]+1:
                    groups[-1].append(slice_idx)
                else:
                    groups.append([slice_idx])

            for group in groups:
                # group contains slices indices of a sub-volume
                ims = load_slices(dir1, group)
                #key_slice = 
                fn_out = out_fmt % (dir1, group[0], group[-1])
                slices2nifti(ims, fn_out, spacings1)
            
            # shutil.rmtree(folder_path)      #Code to delete the unzipped directory once it has been used to convert the files to nifti
            
