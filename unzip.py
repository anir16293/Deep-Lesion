#unzip script

import zipfile
import os

download_directory = ''
unzip_directory = ''

for idx in range(1, 20):
    full_fn = os.path.join(download_directory,'Images_png_%02d.zip' % (idx+1))
    zip_ref = zipfile.ZipFile(full_fn)
    zip_ref.extractall(unzip_directory)
    zip_ref.close()
    os.remove(full_fn)
    print('Unzipped directory '+ str(idx)+ ' and deleted zip file')
