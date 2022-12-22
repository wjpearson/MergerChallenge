import numpy as np
from astropy.table import Table
from astropy.io import fits

import shutil

path = '/mnt/home/wpearson/merger_challenge/TNG/data_tng/'

table = Table.read(path+'catalog_merger_challenge_TNG_test.fits')

test_path = '/mnt/home/wpearson/merger_challenge/TNG/data_tng/test/'

snap_dict = {50: 'snapnum_50_57', 51: 'snapnum_50_57', 52: 'snapnum_50_57', 53: 'snapnum_50_57',
             54: 'snapnum_50_57', 55: 'snapnum_50_57', 56: 'snapnum_50_57', 57: 'snapnum_50_57',
             58: 'snapnum_58_66', 59: 'snapnum_58_66', 60: 'snapnum_58_66', 61: 'snapnum_58_66',
             62: 'snapnum_58_66', 63: 'snapnum_58_66', 64: 'snapnum_58_66', 65: 'snapnum_58_66',
             66: 'snapnum_58_66',
             67: 'snapnum_67_77', 68: 'snapnum_67_77', 69: 'snapnum_67_77', 70: 'snapnum_67_77',
             71: 'snapnum_67_77', 72: 'snapnum_67_77', 73: 'snapnum_67_77', 74: 'snapnum_67_77',
             75: 'snapnum_67_77', 76: 'snapnum_67_77', 77: 'snapnum_67_77',
             78: 'snapnum_78_91', 79: 'snapnum_78_91', 80: 'snapnum_78_91', 81: 'snapnum_78_91',
             82: 'snapnum_78_91', 83: 'snapnum_78_91', 84: 'snapnum_78_91', 85: 'snapnum_78_91',
             86: 'snapnum_78_91', 87: 'snapnum_78_91', 88: 'snapnum_78_91', 88: 'snapnum_78_91',
             89: 'snapnum_78_91', 90: 'snapnum_78_91', 91: 'snapnum_78_91'}

test = 0
for i in range(0, len(table)):
    name = 'snpID_'+str(table[i]['snapnum'])+'_objID_'+str(table[i]['ID'])+'.fits'
    
    try:
        hdu = fits.open(test_path+snap_dict[table[i]['snapnum']]+'/objID_'+str(table[i]['ID'])+'.fits')
    except:
        continue
    if hdu[0].data.shape[0] != 128 or hdu[0].data.shape[1] != 128:
        data = hdu[0].data
        s1 = hdu[0].data.shape[0]
        s2 = hdu[0].data.shape[1]
        data = data[(s1-128)//2:-(s1-128)//2, (s2-128)//2:-(s2-128)//2]
        hdu[0].data = data
    
    hdu.writeto(test_path+'cut_128/'+name)
    test += 1
    hdu.close()
print('test', test)
