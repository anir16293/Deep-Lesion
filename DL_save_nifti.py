#!/usr/bin/env python
"""
Ke Yan
Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
National Institutes of Health Clinical Center
May 2018
THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

"""
A simple demo to load 2D 16-bit slices from DeepLesion and save to 3D nifti volumes.
The nifti volumes can be viewed in software such as 3D slicer and ITK-SNAP.
"""


import numpy as np
import nibabel as nib
import os
import cv2
import csv
import sys
import pandas as pd
import matplotlib.pyplot as plt


dir_in = '/Users/aniruddha/Downloads/CT_files_download/Images_png'
dir_out = '/Users/aniruddha/Downloads/CT_files_nifti'
out_fmt = '%s_%03d-%03d.nii.gz'  # format of the nifti file name to output
# file name of the information file
info_fn = '/Users/aniruddha/Downloads/DL_info.csv'


#dir_in = sys.argv[1]
#dir_out = sys.argv[2]
#info_fn = sys.argv[3]

info_file = pd.read_csv(info_fn)
lesion_mapper = {'bone':1, 'abdomen':2, 'mediastinum': 3,'liver':4, 'lung':5, 'kidney': 6, 'soft_tissue':7, 'pelvis':8}

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
    path_out = os.path.join(dir_out, fn_out)
    nib.save(img, path_out)
    print (fn_out, 'saved')


def load_slices(dir, slice_idxs):
    """load slices from 16-bit png files"""
    slice_idxs = np.array(slice_idxs)
    assert np.all(slice_idxs[1:] - slice_idxs[:-1] == 1)
    ims = []
    key_slice = int((len(slice_idxs)-1)/2)
    for slice_idx in slice_idxs[key_slice - 1: key_slice + 2]:
        fn = '%03d.png' % slice_idx
        path = os.path.join(dir_in, dir, fn)
        dir_path = os.path.join(dir_in, dir)
        len1 = len(slice_idxs)
        im = plt.imread(path, -1)  # -1 is needed for 16-bit image
        assert im is not None, 'error reading %s' % path
        print ('read', path)
        #im = (im.astype(np.int32) - 32768).astype(np.int16)
        #im = ((im - im.min())/(im.max() - im.min()))*255
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
