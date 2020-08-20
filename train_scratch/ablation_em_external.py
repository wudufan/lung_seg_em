#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys
import argparse
import unet2d
import pandas as pd
import SimpleITK as sitk
import scipy.ndimage
import matplotlib.pyplot as plt
import scipy.stats
import imageio


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='/raid/COVID-19/CT-severity/processed/dataset/')
parser.add_argument('--include', type=str, default='medseg_1')

parser.add_argument('--restore_dir', type=str, 
                    default='/raid/COVID-19/CT-severity/results/Iran-2020-04-01-with-annotation/unet2d_256x256x7_mask/em/bias_-8.5_scale_1.75/')
parser.add_argument('--checkpoint', type=str, default='49')
parser.add_argument('--output_file', type=str, default='ablation.csv')

parser.add_argument('--device', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=16)

net = unet2d.unet2d()
parser = net.add_to_parser(parser)


# In[3]:


if sys.argv[0] != 'ablation_em_external.py':
    args = parser.parse_args(['--device', '5', '--n_class', '3'])
else:
    args = parser.parse_args()

for k in vars(args):
    print (k, '=', vars(args)[k])


# In[4]:


# build network
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
tf.reset_default_graph()
model = unet2d.unet2d()
model.from_args(args)
model.build_unet_mask()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# restore
loader = tf.train.Saver()
loader.restore(sess, os.path.join(args.restore_dir, args.checkpoint))


# In[5]:


def load_dataset(input_dir, cohort, exclude_set = []):
    inds = [int(os.path.basename(s)[:-len('.npz')]) for s in glob.glob(os.path.join(input_dir, '*.npz'))]
    inds = [d for d in inds if d not in exclude_set]
    
    if len(inds) == 0:
        return None
    
    dataset = {}
    for index in inds:
        f = np.load(os.path.join(input_dir, '%d.npz'%index))
        for k in f:
            if k not in dataset:
                dataset[k] = []
            dataset[k].append(f[k])
    
    for k in dataset:
        dataset[k] = np.concatenate(dataset[k])
    dataset['cohort'] = np.array([cohort] * len(dataset[dataset.keys()[0]]))
    
    return dataset


# In[6]:


# get studying cohorts
include_cohort = args.include.split(',')
print (include_cohort)
cohorts = [os.path.basename(s) for s in glob.glob(os.path.join(args.input_dir, '*')) if os.path.basename(s) in include_cohort and os.path.isdir(s)]
print (cohorts)


# In[7]:


# loading data
data_list = []
for cohort in cohorts:
    print ('loading %s'%cohort)
    d = load_dataset(os.path.join(args.input_dir, cohort, 'npzs', 'with_unet_pred'), cohort)
    if d is not None:
        data_list.append(d)
dataset = {}
for k in data_list[0]:
    dataset[k] = np.concatenate([d[k] for d in data_list])


# In[8]:


# load patient information
patient_infos = {}
for cohort in cohorts:
    if os.path.exists(os.path.join(args.input_dir, cohort, 'patient_types.npy')):
        info = np.load(os.path.join(args.input_dir, cohort, 'patient_types.npy'), allow_pickle=True).item()
        for k in info:
            info[k]['cohort'] = cohort
        patient_infos.update(info)

# load exclusion
df = []
for cohort in cohorts:
    if os.path.exists(os.path.join(args.input_dir, cohort, 'mrn_train.csv')):
        df.append(pd.read_csv(os.path.join(args.input_dir, cohort, 'mrn_train.csv')))
if len(df) == 0:
    mrn_to_exclude = []
else:
    mrn_to_exclude = list(pd.concat(df, ignore_index=True).filename)
print (mrn_to_exclude)


# In[9]:


def get_patient_data_mrn(mrn, dataset, patient_infos):
    patient = {}
    
    inds = np.where(dataset['mrns'] == mrn)[0]
    
    for k in dataset:
        patient[k] = dataset[k][inds]
    if mrn in patient_infos:
        patient['info'] = patient_infos[mrn]
    else:
        patient['info'] = None
        
    return patient


# In[10]:


def hard_dice(img, label):
    return 2 * np.sum(img * label, dtype = np.float32) / (np.sum(img) + np.sum(label) + 1e-4)


# In[11]:


# predict all
preds = []
imgs = dataset['img']
masks = np.where(dataset['pred'] > 0.5, 1, 0)
print (imgs.shape[0])
for ibatch in range(0, imgs.shape[0], args.batch_size):
    if (ibatch // args.batch_size + 1) % 10 == 0:
        print (ibatch, end=',')
    batch_x = imgs[ibatch:ibatch+args.batch_size]
    batch_mask = masks[ibatch:ibatch+args.batch_size]
    pred = sess.run(model.pred, {model.X: batch_x, model.mask: batch_mask, model.phase: 0})

    preds.append(pred)
print ('concatenating pred')
preds = np.concatenate(preds)

dataset['pred_type'] = preds


# In[12]:


# record the type labels
dataset['pred_final'] = np.copy(dataset['pred'])
dataset['pred_final'][dataset['pred_type'] > 0.5] = 2

th = (1024 - 200) / 110.0
dataset['pred_th'] = np.copy(dataset['pred'])
con_th = np.where(imgs[..., [3]] > th, 1, 0) * masks[...,[0]]
dataset['pred_th'][con_th == 1] = 2


# In[13]:


# calculate dice
label_con = np.where(dataset['label'] == 2, 1, 0)
pred_con = np.where(dataset['pred_final'] == 2, 1, 0)
pred_con_th = np.where(dataset['pred_th'] == 2, 1, 0)
dice_pred_con = hard_dice(label_con, pred_con)
print (dice_pred_con)

# threshold dice
dice_th_con = hard_dice(label_con, con_th)
print (dice_th_con)


# In[14]:


masks = np.where(dataset['label'] > 0, 1, 0)
sensitivity = np.sum(masks * pred_con * label_con, dtype=np.float32) / np.sum(masks * label_con)
sensitivity_th = np.sum(masks * pred_con_th * label_con, dtype=np.float32) / np.sum(masks * label_con)
print (sensitivity, sensitivity_th)

specificity = np.sum((masks-pred_con) * (masks-masks*label_con), dtype=np.float32) / np.sum(masks-masks*label_con)
specificity_th = np.sum((masks-pred_con_th) * (masks-masks*label_con), dtype=np.float32) / np.sum(masks-masks*label_con)
print (specificity, specificity_th)


# In[135]:


import pandas as pd
df = pd.DataFrame({'dice': dice_pred_con, 'sensitivity': sensitivity, 'specificity': specificity}, index=[0])
df.to_csv(os.path.join(args.restore_dir, 'log', args.output_file), index=False)


# In[ ]:




