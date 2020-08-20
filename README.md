## Severity and Consolidation Quantification of COVID-19 from CT Images Using Deep Learning Based on Hybrid Weak Labels
This is the core code for the supervised segmentation of infection and weakly supervised segmentation of the consolidation. EM algorithm was used for the weak-label training, which used only patient-level annotation to train a segmentation network for consolidation regions. The patient-level information is the exsitence of the consolidation in this patient. The good performance was mainly achieved from the pixel intensity prior model built into the EM algorithm. 

The repository is still under construction by cleaning our development code, currently:
1. `/preprocess` is fully cleaned, which provides a pipeline to ready the images from MedSeg
2. `/train_scratch` has not yet been cleaned. But the code provided there are the final training and testing code, which are be refered for the implementation. 
3. `/weights` host the trained checkpoints for unet1 and unet2 that has the best performance. 

We are currently working towards a clean version of the training and testing code. 
