"""
Description: 
    
This file contains an algorithm for taking WSI images and 
converting them to JPEG image patches. These WSI immages only have one
binary label : 1 - there is carcinoma present, 0 - there is no carcinoma present

WSI image: large image with dimensions 10^4 x 10^4. stored in svs format
patch image: can be 10^2 x 10^2, for example 500 x 500 pixels stored in JPEG

The goal is to take image patches from the WSI and store them in the Data folder.
for use in training a CNN.

within the Data folder have the following structure:
    
healthy_tissue:
    Breast:
        train
        val
    Lymph:
        train
        val
        
Citations:
    
    ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
    https://iciar2018-challenge.grand-challenge.org/home/
    
File Info:
Created on Wed Oct 13 22:58:42 2021
@author: Juan Rios
"""

# %% ----- Imports -------------------------------------------------------------
import csv
import os
import openslide
import numpy as np
from PIL import Image

#%% ----- Constants -----------------------------------------------------------
valid_images = ['.svs']
valid_modes = ['np', 'patch']
valid_classes = ['0','1'] # 0 means healthy, 1 means can contain cancer
# %% ----- Hyperparameters -----------------------------------------------------

# change these file paths as necessary
src_data_path = "./data/rawData/axillary lymph/"
dest_train_data_path = "./data/healthy_tissue/lymph/train/"
dest_val_data_path = "./data/healthy_tissue/lymph/val/"
csv_file_name = 'target.csv'
tissue = 'normalLymph'

# the patch size
patch_size = (224, 224)
# how many wsi images to extract from
tot_num_wsi = 140
# if a patch has contains more than this threshold, discard
bckgrnd_ratio = 0.50
white_thresh = 225
black_thresh = 15
trn_per_val = 5 # how many training images per validation

save = True
mode = valid_modes[1]  # 'patch' saves the individual patch as JPEG, 'np' saves the
img_class = valid_classes[0]





#%% ----- background check ----------------------------------------------------
"""
converts the image passed into greyscale image, if the number of white pixels
is higher than the allowable ration, return true, else return false.
white pixels are pixels whose grayscale value is above a certain threshold
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
    
    
    
#%% ----- Import CSV ----------------------------------------------------------
"""
Opens csv file in the specified directory
"""
def csv_to_list(file_path):
    file = open(file_path, "r")
    csv_reader = csv.reader(file)

    lists_from_csv = []
    for row in csv_reader:
        lists_from_csv.append(row)

    return lists_from_csv
    

#%% ----- search list of lists ------------------------------------------------
def get_image_class(targets, img_name):
    
    
    for lst in targets:
        if img_name == lst[0]:
            return lst[1]
    return -1;
    
    
#%% ----- Main Method ---------------------------------------------------------


def main():
    # check if not valid mode
    if mode not in valid_modes:
        print("not a valid mode selected")
        return
    # load target 
    
    targets = csv_to_list(os.path.join(src_data_path, csv_file_name))
    
    
    # For each .svs file in the src folder
    saved_counter = 0;
    wsi_counter = 0;
    for svs_idx, svs_img in enumerate(os.listdir(src_data_path)):
        
        # check if the svs image is the right class
        if get_image_class(targets, svs_img) is not img_class:
            continue

        ext = os.path.splitext(svs_img)[1]

        # conditions to stop the loop
        if ext.lower() not in valid_images:
            continue
        if wsi_counter >= tot_num_wsi :
            return
        curr_path = os.path.join(src_data_path, svs_img)
        print(curr_path)

        # open scan
        scan = openslide.OpenSlide(curr_path)
        
        
        orig_w = np.int(scan.properties.get('aperio.OriginalWidth'))
        orig_h = np.int(scan.properties.get('aperio.OriginalHeight'))

        # create an array to store our image
        if mode == valid_modes[0]:
            np_img = np.zeros((orig_w, orig_h, 3), dtype=np.uint8)

        for row in range(0, orig_w, patch_size[0]):
            for col in range(0, orig_h, patch_size[1]):
                
              
                # conditionals check if within bounds of WSI
                if col + patch_size[1] > orig_h and row + patch_size[0] <= orig_w:
                    p = orig_h - col
                    img = np.array(scan.read_region(
                        (col, row), 0, (p, patch_size[1])), dtype=np.uint8)[..., 0:3]
                elif col + patch_size[1] <= orig_h and row + patch_size[0] > orig_w:
                    p = orig_w - row
                    img = np.array(scan.read_region(
                        (col, row), 0, (patch_size[0], p)), dtype=np.uint8)[..., 0:3]
                elif col + patch_size[1] > orig_h and row + patch_size[0] > orig_w:
                    p = orig_h - col
                    pp = orig_w - row
                    img = np.array(scan.read_region(
                        (col, row), 0, (p, pp)), dtype=np.uint8)[..., 0:3]
                else:
                    img = np.array(scan.read_region(
                        (col, row), 0, (patch_size[0], patch_size[1])), dtype=np.uint8)[..., 0:3]
                # save into np array if 'np'
                if mode == valid_modes[0]:
                    np_img[row:row + patch_size[0],
                           col:col + patch_size[1]] = img
                # save as JPEG if 'patch'
                elif mode == valid_modes[1] and save:
                    jpeg_img = Image.fromarray(img)
                    if check_if_bckgrnd(jpeg_img):
                        continue
                    
                    # if every 5th image, save to validation
                    if saved_counter % 5 == 0:
                        jpeg_img.save(os.path.join(dest_val_data_path, 
                                                   str(saved_counter).zfill(6) +
                                                   '_' + tissue + '.jpeg'))
                    else: 
                        jpeg_img.save(os.path.join(dest_train_data_path, 
                                                   str(saved_counter).zfill(6) +
                                                   '_' + tissue + '.jpeg'))
                    saved_counter  += 1

        if mode == valid_modes[0] and save:
            name_no_ext = os.path.splitext(svs_img)[0]
            np.save(src_data_path + name_no_ext, np_img)

        scan.close
        print('working on WSI: ' + str(wsi_counter))
        wsi_counter +=1


# ----- Call to Main ----------------------------------------------------------
"""
NOTE: This if statement helps guard that you don't create subprocesses recursively
"""
if __name__ == '__main__':
    main()
