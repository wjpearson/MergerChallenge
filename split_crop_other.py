import numpy as np
from astropy.io import fits

import shutil
import glob

path = '/mnt/storage_disk3/mlchallenge/data/data_hsc/'

dicts = ['snapnum_50_57', 'snapnum_58_66', 'snapnum_67_77', 'snapnum_78_91']

test = 0
small = 0
for i in range(0, len(dicts)):
    files = glob.glob(path+dicts[i]+'/*.fits')
    
    for file in files:
        name = file.split('/')[-1]

        try:
            hdu = fits.open(file)
        except:
            continue
        if hdu[0].data.shape[0] < 128 or hdu[0].data.shape[1] < 128:
            print('too small')
            small += 1
        if hdu[0].data.shape[0] != 128 or hdu[0].data.shape[1] != 128:
            data = hdu[0].data
            s1 = hdu[0].data.shape[0]
            s2 = hdu[0].data.shape[1]
            data = data[(s1-128)//2:-(s1-128)//2, (s2-128)//2:-(s2-128)//2]
            hdu[0].data = data

        hdu.writeto(path+'cut_128/'+name)
        test += 1
        hdu.close()
print('test', test)
print('small', small)