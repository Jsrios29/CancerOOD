"""
Description: 
    
This file contains an algorithm for taking WSI images and 
converting them to JPEG image patches. These WSI images come with XML files which
hold the coordinates for annotated regions. This scrip will run through each WSI
and extract patches, and place them into the appropriate folder based on the region


WSI image: large image with dimensions 10^4 x 10^4. stored in svs format
patch image: can be 10^2 x 10^2, for example 500 x 500 pixels stored in JPEG

The goal is to take healthy image patches from the WSI and store them in the Data folder.
for use in training a CNN.

within the Data folder have the following structure:
    
Data:
    healthy_tissue:
        Breast:
            train
            val
    cancer:
        benign
        inSitu
        Invasive
Citations:
    
    ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
    https://iciar2018-challenge.grand-challenge.org/home/
    
File Info:
Created on Wed Oct 13 22:58:42 2021
@author: Juan Rios
"""
#%% ----- Imports -------------------------------------------------------------
# File processing
import csv
import xml.etree.ElementTree as ET
import openslide
# Image processing
import numpy as np
from PIL import Image
import cv2
# other
import os
from datetime import datetime


#%% ----- Constant vars -------------------------------------------------------
valid_ext = ['.svs', '.xml']
cancer_classes = ['benign','in situ', 'invasive'] 
#%% ----- Hyper Parameters ----------------------------------------------------


# change these file paths as necessary
src_path = "./data/raw_data/breast/BACH/labeled/"
cancer_dest = "./data/cancer/"
healthy_dest = {
  "train": "./data/healthy_tissue/breast/train/",
  "val": "./data/healthy_tissue/breast/val/"
}

cancer_dest = {x : os.path.join(cancer_dest, x) for x in cancer_classes}

# the patch size
patch_size = (224, 224)

# if a patch has contains more than this threshold, discard
bckgrnd_ratio = 0.50
white_thresh = 200
black_thresh = 15
cancer_thresh = [0.10, 0.50] # lower and upper bounds
trn_per_val = 5 # how many training images per validation
save = True

#%% ----- Helper Methods ------------------------------------------------------
"""
converts the image passed into greyscale image 

if the number of white pixelsis higher than the allowable ratio
    return true
if the number of black pixels is higher than the allowable ratio
    return true
return false

img - an image to check if its mostly background
"""
def check_if_bckgrnd(img):
    
    gry_img = img.convert('L')
    np_img = np.array(gry_img)
    #num of white pixels
    num_white_pixel = np.count_nonzero([np_img >= white_thresh]) 
    
    # check if mostly white
    if (num_white_pixel / np_img.size) > bckgrnd_ratio:
        return True
    
    num_black_pixel =  np.count_nonzero([np_img <= black_thresh])
    
    if (num_black_pixel / np_img.size) > bckgrnd_ratio:
        return True
    
    return False

"""
Finds all files in a directory with a determined extension

directory - the folder to search
extension - the extension to search for

return files - the list of files with the extension
"""
def findExtension(directory, extension='.xml'):
    files = []    
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
    files.sort()
    return files

"""
gets the coordinates and labels from an '.xml' file and reutns them as a list

filename - the name of the xml file containing the coords

return coords - a list of coords
return labels - labels corresponding to the coords
"""
def read_xml(filename):
    
    tree = ET.parse(filename)
    
    root = tree.getroot()
    regions = root[0][1].findall('Region')
       
    labels = []
    coords = []

    for r in regions:
       
        try:
            label = r[0][0].get('Value')
        except:
            label = r.get('Text')
        if cancer_classes[0] in label.lower():
            label = 1
        elif cancer_classes[1] in label.lower():
            label = 2
        elif cancer_classes[2] in label.lower():
            label = 3
        
        labels += [label]
        vertices = r[1]
        coord = []
        for v in vertices:
            x = int(v.get('X'))
            y = int(v.get('Y'))
            coord += [[x,y]]
    
        coords += [coord]

    return coords,labels

"""
Generates a mask given a list of labels and coordinates

"""

def fill_mask(img_size, coords, labels):
    
    mask  = np.zeros((img_size[0], img_size[1]), dtype=np.uint8)
    
    for coor, lbl in zip(coords, labels):
        cv2.fillPoly(mask,[np.int32(np.stack(coor))],color=lbl)
    
    # This is for debugging    
    # mask2 = mask[::10,::10]
    # mask2 = Image.fromarray(mask2)
    # mask2.show()
    return mask

"""
Determines if the image is cancer, that is if the number of non-zero
elements is greater than the cancer threshold, then the methods returns true

mask - the mask to check if it has cancer
"""
def check_if_cancer(mask, img, counter):
    
    num_nonzero = np.count_nonzero(mask)
    
    if (num_nonzero / mask.size) >= cancer_thresh[1]:
        save_cancer(mask, img, counter)  
        return True;
    elif (num_nonzero / mask.size) >= cancer_thresh[0]:
        return True
    return False

"""
takes a JPEG image with cancer and saves it to the appropriate directory
"""
def save_cancer(mask, img, counter):
    
    counts = np.bincount(mask.flat)
    max_label = np.argmax(counts[1:]) #0 bening, 1 #in situ, 2 invasive
    
    dest = cancer_dest[cancer_classes[max_label]]
    img.save(dest + '/' + str(counter).zfill(6) + '.jpeg')
   
    

"""
This methods takes a WSI, scans through every patch, and saves it in the correct
folder

"""    
def save_patches(mask, scan, img_size):
    
    orig_w = img_size[0]
    orig_h = img_size[1]
    
    saved_counter = 0 # for normal tissue only
    cancer_counter = 0
    
    for row in range(0, orig_w, patch_size[0]):
        for col in range(0, orig_h, patch_size[1]):
            
            # conditionals check if within bounds of WSI
            if col + patch_size[1] > orig_h and row + patch_size[0] <= orig_w:
                p = orig_h - col
                img = np.array(scan.read_region(
                    (col, row), 0, (p, patch_size[1])), dtype=np.uint8)[..., 0:3]
                mini_mask = mask[row : row + patch_size[1], col : col + p]
            elif col + patch_size[1] <= orig_h and row + patch_size[0] > orig_w:
                p = orig_w - row
                img = np.array(scan.read_region(
                    (col, row), 0, (patch_size[0], p)), dtype=np.uint8)[..., 0:3]
                mini_mask = mask[row : row + p, col : col + patch_size[0]]
            elif col + patch_size[1] > orig_h and row + patch_size[0] > orig_w:
                p = orig_h - col
                pp = orig_w - row
                img = np.array(scan.read_region(
                    (col, row), 0, (p, pp)), dtype=np.uint8)[..., 0:3]
                mini_mask = mask[row : row + pp, col : col + p]
            else:
                img = np.array(scan.read_region(
                    (col, row), 0, (patch_size[0], patch_size[1])), dtype=np.uint8)[..., 0:3]
                mini_mask = mask[row : row + patch_size[1], col : col + patch_size[0]]
            
            
            # check if mini mask and img have same size
            if (mini_mask.shape[0] != img.shape[0]) or (mini_mask.shape[1] != img.shape[1]):
                print("Error, minimask and img patch size do not line up")
            
            # save as JPEG if 'patch'
            if save:
                           
                jpeg_img = Image.fromarray(img)
                if check_if_bckgrnd(jpeg_img):
                    continue
                
                if check_if_cancer(mini_mask, jpeg_img, cancer_counter):
                    cancer_counter = cancer_counter + 1
                    continue
                
                # if every 5th image, save to validation
                # if saved_counter % trn_per_val == 0:
                #     jpeg_img.save(os.path.join(healthy_dest['val'], 
                #                                str(saved_counter).zfill(6) +
                #                                 '.jpeg'))
                # else: 
                #     jpeg_img.save(os.path.join(healthy_dest['train'], 
                #                                str(saved_counter).zfill(6) +
                #                                '.jpeg'))
                saved_counter  += 1

    
    
#%% ----- Main ----------------------------------------------------------------

def main():
    
    svs_imgs = findExtension(src_path, valid_ext[0])
    xml_files = findExtension(src_path, valid_ext[1])


    for svs_idx, svs_img in enumerate(svs_imgs):
        
        coords, labels = read_xml(src_path + xml_files[svs_idx])
        
        print('Reading scan',svs_img)
        scan = openslide.OpenSlide(src_path + svs_img)
        dims = scan.dimensions
        wsi_size = (dims[1],dims[0],3) # (H, W, C)
        
        mask = fill_mask(wsi_size, coords, labels)
        save_patches(mask, scan, wsi_size)
        
        print('finished')

#%% ----- Call to Main --------------------------------------------------------

"""
NOTE: This if statement helps guard that you don't create subprocesses recursively
"""
if __name__ == '__main__':
    main()


