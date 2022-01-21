# CancerOOD
A project that uses out-of-distribution detection frameworks to detect abnormal tissue such as breast cancer in breast histopathological images

## Summary
The goal of this project is to use out-of-distribution (OOD) detection frameworks to flag abnormal tissue in histpathological images of of human breast tissue. We train Deep neural networks to identify among healthy breast and healthy lymph node tissue, then we apply OOD detection frameworks, such as energy-based, Outlier-Exposure, and ODIN on breast tissue images with the presence of normal and abornam tissue such as ductal in situ and invasive carcinoma.

## Files
- BMI826_Final_Project.pdf: The final project report submitted. This report details the motivation, methodlogy, and results of this project.
- HelperMethods.py: This module contains an collection of functions that load, train, and test deep learning models.
- OutDetection.py: This module loads a model and a collection of images, then applies OOD detection algorithms to classify each image as either out- or in-distribution.
- PatchExtract.py: This module loads a region of interest (ROI) from an .svs image. The ROI is of pre=-determined size, and the images collected are overlapping patches used as input for OOD detection.
- Patch.Overlay.py: This module loads in an array of scores outputted by Patch.Predict.py for a series of patch images outputted by PatchExtract.py, along with a numpy array that holds the counts of how many times each pixel in an ROI appear in the patches.
-  dpAnnotatedWSI.py: This module contains an algorithm for taking WSI images and converting them to JPEG image patches. These WSI images come with XML files which hold the coordinates for annotated regions. This scrip will run through each WSI and extract patches, and place them into the appropriate folder based on the region.
-  dpBinaryWSI.py: This module contains an algorithm for taking WSI images and converting them to JPEG image patches. These WSI immages only have one binary label : 1, there is carcinoma present, 0, there is no carcinoma present.
- dpMisc: This file contains multiple methods to process the various training data for images in various datasets. This file is simply a bag of scripts to process
the diverse image data formatting in this project.  
- main.py: This module handles the arguement parsin, and calls HelpErMethods.py to train and test a deep learning model. 
