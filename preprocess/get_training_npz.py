# further downsample the images to 256x256, and make the training/testing samples with 7 consecutive slices

import numpy as np
import SimpleITK as sitk
import os
import glob
import scipy.ndimage

import argparse
parser = argparse.ArgumentParser(description = 'generat the npz files for training and testing')
parser.add_argument('--input_dir', type = str)
parser.add_argument('--output_dir', type = str)

parser.add_argument('--size', type = int, default = 256)
parser.add_argument('--nchannels', type = int, default = 7)

def get_patient_type_list(input_dir, output_dir):    
    '''
    generate an indexing npy file to record all the patient types
    '''
    
    filenames = glob.glob(os.path.join(input_dir, 'label/*.nii.gz'))
    filenames.sort(key = lambda f: os.path.basename(f)[:-7])
    # this is to keep the filelist exactly the same with the one during the development
    filenames = [filenames[i] for i in [0,2,1,4,3,6,5,8,7]]
    type_dict = {}
    for filename in filenames:
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        shortname = os.path.basename(filename)[:-7]
        all_types = np.unique(img).astype(int)
        all_types = all_types[all_types > 0]
        type_dict[shortname] = {'all_types': all_types}
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'patient_types'), type_dict)
    
    return type_dict

def convert_3d_to_slices(filename, input_dir, target_shape = [256,256], nchannels = 7):
    '''
    Convert to the volume to slices to get ready for network input.
    params:
    @filename - the short filename to process without dirname or postfix
    @input_dir - master input directory, contains subfolders of ct, lung_mask and label
    '''
    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir, 'ct',filename+'.nii')))
    img = (img + 1024) / 110.0
    img[img > (1024 + 160) / 110.0] = (1024 + 160) / 110.0
    
    lung = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir, 'lung_mask', filename+'.nii')))
    lung = np.where(lung > 0, 1, 0)
    label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(input_dir, 'label', filename+'.nii')))
    
    # flip to make it the same with Iran
#     img = img[:, ::-1, :]
#     lung = lung[:, ::-1, :]
#     label = label[:, ::-1, :]
    
    # zoom
    zoom = [1, target_shape[0] / float(img.shape[1]), target_shape[1] / float(img.shape[2])]
    img = scipy.ndimage.interpolation.zoom(img, zoom, order = 1)
    lung = scipy.ndimage.interpolation.zoom(lung, zoom, order = 0)
    label = scipy.ndimage.interpolation.zoom(label, zoom, order = 0)
    
    # rebin to 7 channels
    img_slices = []
    lung_slices = []
    for i in range(nchannels):
        img_slices.append(img[i:img.shape[0]-nchannels+i+1, ...])
        lung_slices.append(lung[i:lung.shape[0]-nchannels+i+1, ...])
    img = np.array(img_slices).transpose([1,2,3,0])
    lung = np.array(lung_slices).transpose([1,2,3,0])
    label = label[nchannels//2:nchannels//2+img.shape[0], ..., np.newaxis]
    
    return img.astype(np.float32), lung.astype(np.float32), label.astype(np.float32)

def read_annotation(type_dict, nslices, filename):
    '''
    The annotation has 6 digits reserved for various infection types. Only the first two digits are used (GGO and consolidation)
    '''
    all_types = type_dict[filename]['all_types']
    if len(all_types) > 1:
        return np.array([[-1] * 6] * nslices)
    else:
        annotation = np.zeros(6, int)
        annotation[all_types[0]-1] = 1
        return np.array([annotation] * nslices)

if __name__ == '__main__':
    args = parser.parse_args()

    for k in args.__dict__:
        print (k,'=',args.__dict__[k])
    
    print ('Generating patient types...', flush=True)
    patient_types = get_patient_type_list(args.input_dir, args.output_dir)
    for k in patient_types:
        print (k, patient_types[k])
    
    print ('Converting images to npz files...', flush=True)
    list_img = []
    list_lung = []
    list_annotation = []
    list_islice = []
    list_label = []
    list_mrn = []

    nchannels = args.nchannels
    for i, filename in enumerate(patient_types):
        print ('%d'%(i+1), end=',', flush=True)
        # read image
        img, lung, label = convert_3d_to_slices(filename, args.input_dir, target_shape = [args.size, args.size], nchannels = args.nchannels)

        # read annotation
        annotation = read_annotation(patient_types, img.shape[0], filename)

        # retrive only the non-zero slices
        lung_area = lung[..., nchannels//2].sum((1,2))
        valid_slices = np.where(lung_area > 1e-6)[0]

        img = (img * lung)[valid_slices, ...]
        label = (label * lung[..., [nchannels//2]])[valid_slices] 
        lung = lung[valid_slices]
        annotation = annotation[valid_slices]
        mrn = [filename] * len(valid_slices)

        list_img.append(img)
        list_lung.append(lung[..., [nchannels//2]])
        list_label.append(label)
        list_annotation.append(annotation)
        list_islice.append(valid_slices)
        list_mrn.append(mrn)
    print ('')
        
    list_img = np.concatenate(list_img)
    list_lung = np.concatenate(list_lung)
    list_label = np.concatenate(list_label)
    list_annotation = np.concatenate(list_annotation)
    list_islice = np.concatenate(list_islice)
    list_mrn = np.concatenate(list_mrn)

    npz_dir = os.path.join(args.output_dir, 'npzs')
    if not os.path.exists(npz_dir):
        os.makedirs(npz_dir)
    np.savez(os.path.join(npz_dir, '0'), 
             img = list_img, 
             label = list_label, 
             lung = list_lung,
             annotation = list_annotation, 
             islice = list_islice, 
             mrn = list_mrn)
    
    # copy the original CT to the directory for visualization
    print ('Copying the original CT for visualization...', flush=True)
    ct_dir = os.path.join(args.output_dir, 'ct_origin')
    if not os.path.exists(ct_dir):
        os.makedirs(ct_dir)
    for i, filename in enumerate(patient_types):
        print (i, end=',', flush=True)

        ct = sitk.ReadImage(os.path.join(args.input_dir, 'ct', filename+'.nii.gz'))
        sitk.WriteImage(ct, os.path.join(ct_dir, filename+'.nii.gz'))
    print ('')