"""
Created on Wed Oct 13 22:58:42 2021

Description: This file contains multiple methods to process the various training
data for images in various datasets. This file is simply a bag of scripts to process
the diverse image dta formatting in this project.

@author: Juan Rios
"""


#%% ----- Imports -------------------------------------------------------------
import h5py
import os
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torchvision import models
import numpy as np
import shutil
import random
# dataDir = "./data/h5"
# valDatFile = "valData.h5"
# valLabFile = "valLabel.h5"
# valData = h5py.File(os.path.join(dataDir, valDatFile), 'r')
# valData = valData['x']


# testImg = valData[0,:,:,:]

# im = Image.fromarray(testImg)
# im.show()
# im = im.resize((224,224), Image.NEAREST)
# im.show()


#%% ----- Rename the Hymnoptera data ----------------------------------



# data_dir = "./data/hymenoptera_data/test/wasps"
# files = os.listdir(data_dir)


# for index, file in enumerate(files):
#     os.rename(os.path.join(data_dir, file), os.path.join(data_dir, ''.join([str(index) + 'wasps', '.jpg'])))

#%% ----- Process Carnivora data ----------------------------------------------

# dest_dir = "./data/animals2/carnivora/val"
# src_dir = "./data/animals2/carnivora/train"
# files = os.listdir(src_dir)


# for file in files:
#     subfiles = os.listdir(os.path.join(src_dir, file))

#     rng = range(0,30)
#     for sbf in zip(subfiles, rng):
#         #os.rename(os.path.join(data_dir, file, sbf), os.path.join(data_dir ,file, ''.join([str(counter),'_',file[0:5], '.jpg'])))
        
#         shutil.move(os.path.join(src_dir, file, sbf[0]), os.path.join(dest_dir,file, sbf[0]))
        
#%% ------ remove excess lymph node data 
       
# import shutil
# import random
# src = './data/healthy_tissue_surplus/lymph/val'
# dest = './data/healthy_tissue/lymph/val'
# files = os.listdir(src)

# rng = range(0, len(files))
# chosen = random.sample(rng, 4043)

# for chsnFile in chosen:
    
#     file = files[chsnFile]
#     shutil.move(os.path.join(src, file), os.path.join(dest,file))


#%% ----- create fake  --------------------------------------------------------
# from torchvision import datasets
# import numpy

# fake_data = datasets.FakeData(size = 32340)

# dest = './data/fake/'

# for i, img in enumerate(fake_data):
    
#     im = img[0]
    
#     im.save(dest + str(i).zfill(6) + '.jpeg')

#%% ------ Create cancer val set ----------------------------------------------
       
# import shutil
# import random
# src = './data/medical/benign/train/b'
# dest = './data/medical/benign/val/b'
# files = os.listdir(src)

# rng = range(0, len(files))
# chosen = random.sample(rng, 4043)

# for chsnFile in chosen:
    
#     file = files[chsnFile]
#     shutil.move(os.path.join(src, file), os.path.join(dest,file))
    
#%% ----- manipulate imagenet data --------------------------------------------

# src = 'E:/Downloads/ILSVRC/Data/CLS-LOC/train/'
# dest = 'C:/Users/JuanRios/.spyder-py3/CS762_Proj/data/auxiliary/imagenet/'
# files = os.listdir(src)
# desired_size = 32400
# num_classes = len(files)
# files_per_class = round(desired_size / num_classes)
# for idx, file in enumerate(files):
#     subfile = os.listdir(os.path.join(src, file))
#     for img in zip(subfile, range(0, files_per_class)):
#         shutil.move(os.path.join(src, file, img[0]), os.path.join(dest,img[0]))
        
    
   # os.rename(os.path.join(src, file), os.path.join(src ,str(idx)))
   
#%% ----- Create 19 Tissue Data -------------

# src = r'E:\Downloads\panuke\Images'
# dest = './data/medical/healthy_tissue_2'
# img_path = 'images2.npy'
# type_path = 'types2.npy'
# images = np.load(os.path.join(src, img_path))
# types = np.load(os.path.join(src, type_path))

# num_imgs = len(types)
# ratio = 5

# for im in range(num_imgs):
    
#     curr_img = images[im, :,:,:]
#     label = types[im]
    
    
    
#     if im % ratio == 0:
#         path = 'val'
#     else:
#         path = 'train'
        
#     if os.path.isdir(os.path.join(dest, path, label)) == False:
#          os.mkdir(os.path.join(dest, path, label))
        
#     jpeg = Image.fromarray(np.uint8(curr_img))  
#     jpeg.save(os.path.join(dest, path, label ,str(im).zfill(6) + '_' +label + '.jpeg' ))
    
    
#%% ------ Sample 300 breast images

src = './data/medical/healthy_tissue/train/breast'
dest = './data/medical/healthy_tissue_2'
files = os.listdir(src)
rng = range(0, len(files))
chosen = random.sample(rng, 400)
counter = 0
for i in chosen:
    source = os.path.join(src, files[i])
    
    if counter < 300:
        path = 'train/Breast'
    else:
        path = 'val/Breast'
    destination = os.path.join(dest, path, files[i])
    shutil.copyfile(source, destination)
    counter += 1
