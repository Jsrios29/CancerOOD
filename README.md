# CancerOOD
A project that uses out-of-distribution detection frameworks to detect abnormal tissue such as breast cancer in breast histopathological images

## Summary
The goal of this project is to use out-of-distribution (OOD) detection frameworks to flag abnormal tissue in histpathological images of of human breast tissue. We train Deep neural networks to identify among healthy breast and healthy lymph node tissue, then we apply OOD detection frameworks, such as energy-based, Outlier-Exposure, and ODIN on breast tissue images with the presence of normal and abornam tissue such as ductal in situ and invasive carcinoma.

## Files
- BMI826_Final_Project.pdf: The final project report submitted. This report details the motivation, methodlogy, and results of this project.
- HelperMethods.py: This module contains an collection of functions that load, train, and test deep learning models.
- OutDetection.py: This model loads a model and a collection of images, then applies OOD detection algorithms to classify each image as either out- or in-distribution.
- PatchExtract.py: This model loads a region of interest (ROI) from an .svs image. The ROI is of pre=-determined size, and the images collected are overlapping patches used as input for OOD detection
