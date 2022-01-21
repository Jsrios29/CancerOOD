
"""
Created on Wed Dec  8 18:56:44 2021

Description: This file loads in an array of scores for a series of patch images,
along with a numpy array that holds the counts of how many times each pixel in an ROI
appear in the patches 
    
@author: Juan Rios
"""

#%% Import Statements
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import cm
import os
import torch

from PatchExtract import regions
from PatchExtract import dest as src
from PatchExtract import stride
from PatchExtract import patch_size
#%% Hyper Params
"""
modes = [('e', -3.58), # ood if greater than
         ('oe', -0.9572),#ood if greater than
         ('o', 0.501) ] # ood if greater than
"""

#modes = [('e', -3.58)]
#modes = [('oe', -0.9572)]

modes = [('e', -3.58),
         ('oe', -0.9572)]
#%% Helper Methods

def load_scores(svs, mode):
    
    if mode == 'e':
        file = 'energy_scores.pt'
    elif mode == 'o':
        file = 'odin_scores.pt'
    elif mode == 'oe':
        file = 'oe_scores.csv'
        
    path = os.path.join(src, svs, file)
    
    if mode == 'e':
        scores = torch.load(path)
    elif mode == 'o':
        scores = 'odin_scores.pt'
    elif mode == 'oe':
        dframe = pd.read_csv(path, header = None)
        scores = torch.tensor(dframe.values)
        scores = scores.squeeze(1)
       
    return scores

def load_counts(svs):
    path = os.path.join(src, svs, 'counts.npy')
    counts = np.load(path)    
    return torch.from_numpy(counts)

def get_heat_map(scores, counts):
    
    height = counts.shape[0]
    width = counts.shape[1]
    
    heat_map = torch.zeros((height, width))
    
    counter = 0
    for row in range(0, height - patch_size[0] , stride):
        for col in range(0, width - patch_size[1], stride):
            
            heat_map[row:row + patch_size[0], col:col + patch_size[1]] += scores[counter].item()
            counter += 1
    
    return torch.div(heat_map, counts)

def maps_to_img(heat_map):
    
    max_val = 255
      
    min_score = torch.min(heat_map)
    max_score = torch.max(heat_map)
    mean_score = torch.mean(heat_map)
    std_score = torch.std(heat_map)
    upper_limit = max_score
    
             
    diff = abs(min_score.item()) - abs(upper_limit.item())
    ratio = max_val/diff
    heat_map = -1*ratio*heat_map + ratio*upper_limit.item()
    heat_map= np.uint8(heat_map)
    img = Image.fromarray(np.uint8(255*cm.spring(heat_map)))
    return img
        
    #map_sum = torch.sum(maps, dim = 0)
   # min_val = torch.min(map_sum).item()
    #alpha = max_val/abs(min_val)
    
   # img = Image.fromarray(np.uint8(alpha*cm.binary(map_sum + abs(min_val))))  

   # std_score = torch.std(_map)
    #upper_limit = mean_score + 3*std_score
    """
    debug
    """
    #min_score = torch.as_tensor([-30])
    #upper_limit = torch.as_tensor([-10])
    """"""
    
    #diff = abs(min_score.item()) - abs(upper_limit.item())
   # ratio = 255/diff
   # heat_img = -1*ratio*_map + ratio*upper_limit.item()
    
    #gray_img = Image.fromarray(np.uint8(_map*255))
    
    #heat_img = np.uint8(heat_img)
    #test = cm.seismic(heat_img)
   # color_img = Image.fromarray(np.uint8(cm.seismic(heat_img)*255))
   

def get_mask(_map, mode):
    
    mod = mode[0]
    t = mode[1]
    mask = _map.clone().detach()
   
    
    if mod == 'e':
        mask[mask >= t] = 1
        mask[mask < t] = 0
    
    elif mod == 'oe':
        mask[mask >= t] = 1
        mask[mask < t] = 0
    elif mod == 'o':
        mask[mask > t] = 1
        mask[mask <=t] = 0
    return mask     

def mask_to_img(mask):
    
    max_val = 255
    alpha = max_val/torch.max(mask).item()
    gray_img = Image.fromarray(np.uint8(alpha*mask))
    return gray_img
    
        
#%% Main

def main(): 
    
    for i, roi in enumerate(regions):
        
        svs = roi[0] + '_' + str(i).zfill(3)
        counts = load_counts(svs) + 0.0001
        
        maps = torch.zeros(( len(modes), counts.shape[0], counts.shape[1]))
        mask_sum = torch.zeros((counts.shape[0], counts.shape[1]))
        
        real_img = Image.open(os.path.join(src, svs, 'real_img.png')).convert('RGBA')
        
        for j, mode in enumerate(modes):
            
            scores = load_scores(svs, mode[0])   
            heat_map = get_heat_map(scores, counts)  
            heat_mask = maps_to_img(heat_map)
         
            mask = get_mask(heat_map, mode)
            mask_sum = torch.add(mask_sum, mask)
            
            heat_mask.putalpha(128)
            heat_overlay = Image.alpha_composite(real_img, heat_mask.convert('RGBA'))
            heat_overlay.save(os.path.join(src, svs, 'heat_img' + mode[0] + '.png'))
    
        
        mask_img = mask_to_img(mask_sum)
        mask_img.putalpha(128)
        overlay_img = Image.alpha_composite(real_img, mask_img.convert('RGBA'))
        overlay_img.save(os.path.join(src, svs, 'bin_mask_img.png'))
        
    
#%% Call to Main

"""
NOTE: This if statement helps guard that you don't create subprocesses recursively
"""
if __name__ == '__main__':
  main()