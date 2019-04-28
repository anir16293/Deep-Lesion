import numpy as np
#import matplotlib.pyplot as plt
import sys
import pandas as pd
import cv2
import os

"""
Command: python3 Deep-Lesion/preprocess.py /media/parv/Seagate\ Backup\ Plus\ Drive/DeepL_Dataset/Extracted\ /1/Images_png/ /home/parv/DL/DL\ Project/Final_Images Deep-Lesion/DL_info.csv
"""

download_path =  sys.argv[1]        # Folder containing the unzipped images
save_path = sys.argv[2]            # Folder to save images

dl_info_csv = pd.read_csv(sys.argv[3])          # The DL_info.csv file path
key_slices = dl_info_csv.File_name.values

if not os.path.isdir(save_path):
    os.makedirs(save_path, 777)
    print('Directory created')

relu = lambda x: x if x > 0 else 0
relu_vector = np.vectorize(relu)

def preprocess_image(im):
    im = (im.astype(np.int32) - 32768).astype(np.int16)
    im = ((im + 1024)/(1024 + 3071))*255
    im = relu_vector(im)
    im = im.astype(np.int8)
    return(im)

for _slice in key_slices:
    slice_name = _slice.split('_')
    image_number = slice_name[-1]
    previous = '.'.join([str(int(image_number.split('.')[0]) - 1).zfill(3), 'png'])
    _next = '.'.join([str(int(image_number.split('.')[0]) + 1).zfill(3), 'png'])
    folder_number = slice_name[0:-1]
    path = '_'.join(folder_number)
    final_path = os.path.join(download_path, path, image_number)
    previous_path = os.path.join(download_path, path, previous)
    next_path = os.path.join(download_path, path, _next)

    try:
        im1 = cv2.imread(final_path, -1)
        im1 = preprocess_image(im1)
    except AttributeError:
        continue
        #print(final_path + ' not available')

    try:
        imp = cv2.imread(previous_path, -1)
        imp = preprocess_image(imp)
    except AttributeError:
        print('Error')
        imp = im1

    try:
        imn = cv2.imread(next_path, -1)
        imn = preprocess_image(imn)
    except AttributeError:
        print('Error')
        imn = im1

    new_img = cv2.merge([imp, im1, imn])
    save_img_name = '_'.join([path, image_number])
    save_image_path = os.path.join(save_path, save_img_name)
    print('File:')
    print(final_path)
    print(previous_path)
    print(next_path)
    cv2.imwrite(save_image_path, new_img)
