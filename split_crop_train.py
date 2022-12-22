import numpy as np
from astropy.table import Table
from astropy.io import fits

import shutil

path = '/mnt/home/wpearson/merger_challenge/TNG/data_tng/'

table = Table.read(path+'catalog_merger_challenge_TNG_train.fits')
valid_sequenceID = np.genfromtxt(path+'valid_sequenceID.txt', dtype=int)

ttl = 0
for sID in valid_sequenceID:
    in_sequence = np.where(table['sequence_ID'] == sID)[0]
    ttl += len(in_sequence)

train_path = '/mnt/home/wpearson/merger_challenge/TNG/data_tng/train/'

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

train = 0
valid = 0
for i in range(0, len(table)):
    name = ''
    for col in table.columns:
        if col == 'time_before_merger' or col == 'time_after_merger':
            name += str(round(table[col][i], 3)).replace('.', 'p')+'.'
        elif col == 'is_major_merger':
            if table[col][i] == 1:
                name += 'M.'
            else:
                name += 'N.'
        elif col == 'is_pre_merger':
            if table[col][i] == 1:
                name += 'PR.'
            elif table['is_ongoing_merger'][i]:
                name += 'OG.'
            elif table['is_post_merger'][i]:
                name += 'PO.'
            else:
                name += 'NM.'
        elif col == 'is_ongoing_merger':
            continue
        elif col == 'is_post_merger':
            continue
        else:
            name += str(table[col][i]).replace('.', 'p')+'.'
    name += 'fits'
    
    try:
        hdu = fits.open(train_path+snap_dict[table[i]['snapnum']]+'/objID_'+str(table[i]['ID'])+'.fits')
    except:
        continue
    if hdu[0].data.shape[0] != 128 or hdu[0].data.shape[1] != 128:
        data = hdu[0].data
        s1 = hdu[0].data.shape[0]
        s2 = hdu[0].data.shape[1]
        data = data[(s1-128)//2:-(s1-128)//2, (s2-128)//2:-(s2-128)//2]
        hdu[0].data = data
    
    if table[i]['sequence_ID'] in valid_sequenceID:
        hdu.writeto(train_path+'cut_128/valid/'+name)
        valid += 1
    else:
        hdu.writeto(train_path+'cut_128/train/'+name)
        train += 1
    hdu.close()
print('valid', valid)
print('train', train)
