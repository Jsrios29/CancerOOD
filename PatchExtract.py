"""
Created on Mon Dec  6 11:01:15 2021

Description: this file loads an svs region of interest. takes a begin corner and
end corner, and collects a series of overlapping patches with stride s. The patches are saved
into a 4D tensor, of size n x c x h x w, where c is the number of channels, h is height
w is width, n and is the number of patches

@author: Juan Rios
"""

#%% ----- Import Statements -----
import openslide
from PIL import Image
import numpy as np
import math
import os
import shutil
#%% ----- Hyper Params -----

patch_size = (224, 224)
src = './data/raw_data/breast/Post-NAT-BRCA/'
dest = './data/raw_data/eval/'
#regions = [('99788', 8140, 16480, 2241, 2241),
           #('99790', 8140, 16480, 2241, 2241)]
regions = [('99797', 10000, 25000, 2*2240 + 1, 1*2240 +1),
           ('99797', 11800, 13600, 2*2240 + 1, 1*2240 + 1)]
"""
This tuple dictates ROI in the form: (top left col, top left row,num cols, num rows)
"""

roi = (26000, 12000, 2241, 2241)
stride = 56

#%% ----- Helper Methods -----


def load_svs(path, file_name):
    scan = openslide.OpenSlide(path + file_name[0:5] +'.svs')
    return scan

def get_roi(scan, roi):
    
    img = np.array(scan.read_region(roi[0:2],0, roi[2:4]),dtype=np.uint8)[...,0:3]
    return img

def extract_patches(roi_arr, stride, svs):
    
    height, width = roi_arr.shape[0:2]
    num_col = math.ceil((width - patch_size[1])/stride)
    num_row = math.ceil((height - patch_size[0])/stride)
    num_patches = num_col * num_row
    patches  = np.zeros((num_patches, 3, patch_size[0], patch_size[1]), dtype=int)
    counts = np.zeros((roi_arr.shape[0:2]), dtype=int)
    patch_counter = 0
    for row in range(0, height - patch_size[0] , stride):
        for col in range(0, width - patch_size[1], stride):
        
            
            patch = roi_arr[row:row + patch_size[0], col:col + patch_size[1], : ] 
            jpeg_patch = Image.fromarray(patch)
            jpeg_patch.save(dest + svs + '/images/' +  str(patch_counter).zfill(6) + '.jpeg')
            
            patch = np.transpose(patch, (2, 0, 1))
            patches[patch_counter, :, :, :] = patch
            counts[row:row + patch_size[0], col:col + patch_size[1]] += 1 
            patch_counter += 1
            
    return patches, counts

def vis_counts(counts):
    
    max_val = np.max(counts)
    scale = (255/max_val)
    counts = scale*counts
    counts_img = Image.fromarray(counts)
    counts_img.show()
    
def save_data(patches, counts, svs):
    path = dest + svs + '/'
    np.save(path + 'counts', counts)
    np.save(path + 'patches', patches)
    
    
def reset_dir(svs): 
    
    if os.path.isdir(dest + svs) == False:
        os.mkdir(dest + svs)
        os.mkdir(dest + svs + '/images')
    else:
        shutil.rmtree(dest + svs)
        os.mkdir(dest + svs)
        os.mkdir(dest + svs + '/images')
#%% ----- Main -----

def main():

    for i, roi in enumerate(regions):
        svs = roi[0] + '_' + str(i).zfill(3)
        reset_dir(svs)
        scan = load_svs(src, svs)
        roi_arr = get_roi(scan, roi[1:])
    
        patches, counts = extract_patches(roi_arr, stride, svs)
    
        save_data(patches, counts, svs)
        #vis_counts(counts)
    
        roi_img = Image.fromarray(roi_arr)
        roi_img.save(os.path.join(dest, svs, 'real_img.png'))
    
    
#%% ----- Call to Main -----

"""
NOTE: This if statement helps guard that you don't create subprocesses recursively
"""
if __name__ == '__main__':
  main()