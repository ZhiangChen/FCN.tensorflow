import os
import numpy as np
from PIL import Image, ImageOps
import glob

data_dir = 'Data_zoo/'

image_list = glob.glob(data_dir + 'images/training/' + '*.png')
# print(image_list)

for fname in image_list:
    print(fname)
    aname = data_dir + 'annotations/training/label_' + fname.split('/')[-1]
    print(aname)
    img = Image.open(fname)
    anno = Image.open(aname)
#     img_arr = np.asarray(img)
    
    img_flip = ImageOps.flip(img)
    anno_flip = ImageOps.flip(anno)
    img_mirror = ImageOps.mirror(img)
    anno_mirror = ImageOps.mirror(anno)
    
    img_flip.save(fname[:-4] + '_f' + '.png')
    anno_flip.save(aname[:-4] + '_f' + '.png')
    img_mirror.save(fname[:-4] + '_m' + '.png')
    anno_mirror.save(aname[:-4] + '_m' + '.png')