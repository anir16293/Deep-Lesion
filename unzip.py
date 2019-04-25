"""
This code is a part of the DeepLesion course project for the Deep Learning course (CS 682), Spring 2019 at John's Hopkins University.
The project team consists of Aniruddha Tamhane, Parv Saxena and Wei-Lun Huang.
The course was taught by Matthias Unberath and TA'd by Jie-Ying Wu and Gao Cong.
The project was sponsored by Google Cloud and Intuitive Surgicals.
"""

import zipfile
import os
import sys

download_directory = str(sys.argv[1])  # Directory containing the zipped image files
unzip_directory = str(sys.argv[2])     # Directory to unzip the zipped image files

for idx in range(1, 20):
    full_fn = os.path.join(download_directory,'Images_png_%02d.zip' % (idx+1))
    zip_ref = zipfile.ZipFile(full_fn)
    zip_ref.extractall(unzip_directory)
    zip_ref.close()
    #os.remove(full_fn)
    print('Unzipped directory '+ str(idx))
