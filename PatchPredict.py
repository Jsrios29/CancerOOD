"""
Created on Mon Dec  6 19:27:16 2021

Description: this file loads a model and loads a series of patches
and gives energy scores for each patch, the scores are then saved as numpy
arrays

@author: Juan Rios
"""

#%% Imports

import numpy as np
import os
import torch
import time
from HelperMethods import initialize_model
from torchvision import datasets, transforms

from PatchExtract import regions
from PatchExtract import dest as src

#%% Hyper params

data_dir = src
model_name = "wide" 
models_dir = './models'
model_Id = 'e2t7.22'
num_classes_in = 2
batch_sz = 15
T = 1
#%% Helper Methods

def collect_patches(input_size, svs):
    
    # initializing the transforms
    data_transforms =  transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    # initializing the dataset    
    print("Initializing datasets and dataloader...")
    
    dataset = datasets.ImageFolder(os.path.join(src, svs), 
                      data_transforms)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                        batch_size= batch_sz, shuffle=False, 
                        num_workers=4) 
    
    return dataloader

def get_energy(model, images, device, T):
    
    print("Getting Energy Scores...")
    since = time.time()

    model = model.to(device)
    model.eval()   # Set model to evaluate mode

    energy_scores = torch.zeros(len(images.dataset.imgs))
    curr_batch = 0

    # feed forward inputs
    for img_batch, _ in images:
        
        batch_size = len(img_batch)
        
        img_batch = img_batch.to(device)

        outputs = model(img_batch)
        _, preds = torch.max(outputs, 1)
        energy_scores[curr_batch: curr_batch + batch_size] = (-T*torch.logsumexp(outputs/T, dim=1)).data 
        curr_batch = curr_batch + batch_size
        
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
           time_elapsed // 60, time_elapsed % 60))
    
    return energy_scores

#%% Main

def main():
    
    # Load the trained model
    model_ft, input_size = initialize_model(model_name, num_classes_in, 
                                            False, False)
                                                           
    # Load the dictionary of paramerters
    model_dict = torch.load(os.path.join(models_dir, model_Id))
    # Set the untrained model params to the loaded
    model_ft.load_state_dict(model_dict)
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    
    for i, roi in enumerate(regions):
        svs = roi[0] + '_' + str(i).zfill(3)
       # Collect the images
        images = collect_patches(input_size, svs)
    
        # Get energy scores
        energy_scores = get_energy(model_ft, images, device, T)
        torch.save(energy_scores, os.path.join(src, svs, 'energy_scores.pt'))
    
#%% Call to Main

"""
NOTE: This if statement helps guard that you don't create subprocesses recursively
"""
if __name__ == '__main__':
  main()