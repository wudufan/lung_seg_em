# resample the medseg dataset to 512x512x5mm, to keep align with our other dataset. Then eveything is further resampled to 256x256 to feed into the network.

import numpy as np
import SimpleITK as sitk
import os
import sys
import matplotlib.pyplot as plt
import scipy.ndimage
import glob
from multiprocessing import Pool
from functools import partial

import argparse
parser = argparse.ArgumentParser(description = 'resample medseg dataset')
parser.add_argument('--input_dir', type = str)
parser.add_argument('--output_dir', type = str)

parser.add_argument('--interpolation', type = str, default = 'linear')
parser.add_argument('--spacing_z_mm', type = float, default = 5)

def resample_single(filename, output_dir, spacing_z_mm = 5, interpolation = 'linear'):
    '''
    resample a single image from medseg and write to the output folder
    '''
    img = sitk.ReadImage(filename)
        
    # calculate zoom factors
    size = img.GetSize()
    spacing = img.GetSpacing()
    # only resample those with slice thickness <=2mm to 5mm slice thickness
    if spacing[-1] <= 2:
        new_spacing = [spacing[0] * size[0] / 512, spacing[1] * size[1] / 512, spacing_z_mm]
    else:
        new_spacing = [spacing[0] * size[0] / 512, spacing[1] * size[1] / 512, spacing[2]]
    
    # Gaussian filter before downsampling
    if interpolation == 'linear':
        sigma = [0.01, 0.01, spacing_z_mm / 4]
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigma)
        img = gaussian.Execute(img)

    # downsampling
    sitk_interp_dict = {'nearest': sitk.sitkNearestNeighbor, 'linear': sitk.sitkLinear}
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk_interp_dict[interpolation])
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    new_size = [512, 512, int(np.ceil(img.GetSize()[2] * spacing[2] / new_spacing[2]))]
    resample.SetSize(new_size)
    img = resample.Execute(img)
    
    # The flip filters are applied to keep it consistent with our other dataset
    flip_filter = sitk.FlipImageFilter()
    flip_filter.SetFlipAxes([False, True, True])
    img = flip_filter.Execute(img)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sitk.WriteImage(img, os.path.join(output_dir, os.path.basename(filename)))

if __name__ == '__main__':
    args = parser.parse_args()

    for k in args.__dict__:
        print (k,'=',args.__dict__[k])
        
    filenames = glob.glob(os.path.join(args.input_dir, '*.nii.gz'))
    p = Pool(8)
    p.map(partial(resample_single, output_dir = args.output_dir, spacing_z_mm = args.spacing_z_mm, interpolation = args.interpolation), filenames)
    p.close()
    