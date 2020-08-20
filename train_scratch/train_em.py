#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys
import argparse
import unet2d
import augmentor2d
import pandas as pd


# In[16]:


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, 
                    default='/raid/COVID-19/CT-severity/processed/Iran-2020-04-01-with-annotation/256x256x7channels_preprocessed_with_mask/weak/')
parser.add_argument('--exclude', type=int, nargs='+', default=[0,100])
parser.add_argument('--output_dir', type=str, 
                    default='/raid/COVID-19/CT-severity/results/Iran-2020-04-01-with-annotation/unet2d_256x256x7_mask/em/')
parser.add_argument('--tag', type=str, default='debug')
parser.add_argument('--restore_dir', type=str, 
                    default='/raid/COVID-19/CT-severity/results/Iran-2020-04-01-with-annotation/unet2d_256x256x7_mask/segmentation/focal_0_area_0/')
parser.add_argument('--checkpoint', type=str, default='199')

parser.add_argument('--device', type=str, default='0')
parser.add_argument('--nepochs', type=int, nargs=1, default=[50], help = 'training of the last conv layer, training of fine tune layer, training of all layer')
parser.add_argument('--lr', type=float, nargs=1, default=[0.0005])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--finetune_layers', type=int, default=9)
parser.add_argument('--balance', type=int, default=1)

parser.add_argument('--bias', type=float, default=-7)
parser.add_argument('--scale', type=float, default=0.5)

net = unet2d.unet2d()
parser = net.add_to_parser(parser)

aug = augmentor2d.multi_thread_augmentor()
parser = aug.add_to_parser(parser)


# In[17]:


if sys.argv[0] != 'train_em.py':
    # background, GGO, consolidation
    args = parser.parse_args(['--device', '1', '--n_class', '3', '--bias', '-7.5', '--scale', '0.5'])
else:
    args = parser.parse_args()

for k in vars(args):
    print (k, '=', vars(args)[k])


# In[18]:


# build network
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
tf.reset_default_graph()
model = unet2d.unet2d()
model.from_args(args)
model.build_unet_mask()
total_loss = model.dice_loss

all_unet_layers = model.down_layer_vars + model.up_layer_vars
all_unet_vars = list(np.concatenate(all_unet_layers))
finetune_unet_vars = list(np.concatenate(all_unet_layers[-args.finetune_layers:]))
new_vars = [v for v in tf.trainable_variables() if v not in all_unet_vars]
bn_vars = [v for v in tf.global_variables() if v not in tf.trainable_variables()]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
lr = tf.placeholder(tf.float32, name = 'lr')
trainers = []
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(lr)
#     trainers.append(optimizer.minimize(total_loss, var_list = new_vars))
#     trainers.append(optimizer.minimize(total_loss, var_list = new_vars + finetune_unet_vars))
    trainers.append(optimizer.minimize(total_loss, var_list = new_vars + all_unet_vars))
    
saver = tf.train.Saver(max_to_keep=args.nepochs[-1])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# restore
loader = tf.train.Saver()
loader.restore(sess, os.path.join(args.restore_dir, args.checkpoint))


# In[19]:


def clean_data(annotations, imgs, labels, args):
    # remove the -1 in the annotations
    sum_annotations = annotations.sum(1)
    inds_include = np.where(sum_annotations >= 0)[0]
    
    # leave only the first two annotations columns
    annotations = annotations[:, :2]
    
    # discard the ones with label but no annotation, or with annotation but no label
    sum_labels = np.sum(labels, (1,2,3))
    sum_annotations = np.sum(annotations, 1)
    inds_exclude_1 = np.where((sum_labels > 0) & (sum_annotations == 0))[0]
    inds_exclude_2 = np.where((sum_labels == 0) & (sum_annotations > 0))[0]
    
    # select the training cohort
    inds = [i for i in inds_include if i not in np.concatenate((inds_exclude_1, inds_exclude_2))]
    
    return annotations[inds], imgs[inds], labels[inds]

def prepare_label_and_annotations(annotations, labels, imgs, th = (1024-200)/110, dice_weight = 1):
    '''
    only label the consolidation
    '''
    label_con = np.copy(labels)
    classes = np.zeros(len(annotations), np.float32)
    # air
    inds = np.where(annotations.sum(1) == 0)[0]
    classes[inds] = 0
    label_con[inds, ...] = 0
    # GGO
    inds_ggo = np.where(annotations[:, 0] == 1)[0]
    classes[inds_ggo] = 1
    label_con[inds_ggo, ...] = 0
    # GGO + consolidation    
    inds_mix = np.where(annotations[:, 1] == 1)[0]
    for i in inds_mix:
        img_mix = imgs[i, ..., 3]
        label_con[i, img_mix < th, 0] = 0     # only preserve larger than threshold
    classes[inds_mix] = 2
    
    return classes, label_con, inds_ggo, inds_mix


# In[20]:


def get_sample_index(classes, balance = True):
    if not balance:
        inds = np.arange(len(classes))
        np.random.shuffle(inds)
        return inds
    
    cnt_per_class = [np.count_nonzero(classes == s) for s in np.unique(classes)]
    target_cnt_per_class = max(cnt_per_class)
    sample_inds = []
    for iclass in range(len(cnt_per_class)):
        base_inds = list(np.where(classes == iclass)[0])
        n_rep = int(np.ceil(target_cnt_per_class / float(len(base_inds))))
        inds = []
        for i in range(n_rep):
            np.random.shuffle(base_inds)
            inds += base_inds
        sample_inds.append(inds[:target_cnt_per_class])
    
    return np.array(sample_inds).T.flatten()


# In[21]:


# load training data
list_filename = glob.glob(os.path.join(args.input_dir, '*.npz'))
imgs = []
labels = []
annotations = []
for filename in list_filename:
    dataset = os.path.basename(filename)[:-len('.npz')]
    if int(dataset) in args.exclude:
        continue
    f = np.load(filename)
    
    imgs.append(f['img'])
    labels.append(f['label'])
    annotations.append(f['annotation'])

imgs = np.concatenate(imgs)
labels = np.concatenate(labels)
annotations = np.concatenate(annotations)

annotations, imgs, labels = clean_data(annotations, imgs, labels, args)
classes, labels_con, inds_ggo, inds_mix = prepare_label_and_annotations(annotations, labels, imgs)
inds_one_type = np.array([i for i in np.arange(len(labels)) if i not in inds_mix])
labels = np.concatenate((labels_con, labels), -1)


# In[22]:


def add_to_summary(summary_writer, loss_vals, loss_names, global_step):
    summary = tf.Summary()
    for val, name in zip(loss_vals, loss_names):
        summary.value.add(tag = name, simple_value = val)
    summary_writer.add_summary(summary, global_step)

def add_to_record(records, loss_vals, loss_names, phase):
    for val, name in zip(loss_vals, loss_names):
        tag = '%s_%s'%(name, phase)
        if not tag in records:
            records[tag] = [val]
        else:
            records[tag].append(val)
    return records


# In[23]:


# load validation dataset
valid_path = '/raid/COVID-19/CT-severity/processed/dataset/medseg_1/npzs/with_unet_pred/0.npz'
with np.load(valid_path) as f:
    valid_imgs = f['img']
    valid_labels = f['label']
    valid_preds = np.where(f['pred'] > 0.5, 1, 0)
    valid_lungs = f['lung']


# In[24]:


def evaluate_dice(sess, model, imgs, labels, masks, args):
    preds = []
    
    for ibatch in range(0, imgs.shape[0], args.batch_size):
        batch_x = imgs[ibatch:ibatch+args.batch_size]
        batch_mask = masks[ibatch:ibatch+args.batch_size]
        preds.append(sess.run(model.pred, {model.X: batch_x, model.mask: batch_mask, model.phase: 0}))
    preds = np.concatenate(preds)
    preds_con = np.where(preds > 0.5, 1, 0)
    labels_con = np.where(labels == 2, 1, 0)
    
    dice_val = 2 * np.sum(preds_con * labels_con, dtype = np.float32) / (np.sum(preds_con) + np.sum(labels_con))
    
    return dice_val


# In[25]:


# tensorboard
summary_dir = os.path.join(args.output_dir, args.tag, 'log')
datasets = ['train']
writers = {}
for dataset in datasets:
    writers[dataset] = tf.summary.FileWriter(os.path.join(summary_dir, dataset), sess.graph)


# In[26]:


def get_label(preds, imgs, masks, class_labels, bias, scale):
    # prior probabilities of being consolidation
    if scale <= 0:
        pred_cons = bias
    else:
        pred_cons = 1 / (1 + np.exp(-(imgs[..., [3]] + bias) * scale))
    pred_cons *= masks
    preds += pred_cons * 1
    preds[class_labels != 2, ...] = 0
    
    return np.where(preds > 0.5, 1, 0).astype(np.float32)


# In[27]:


loader = tf.train.Saver()
loader.restore(sess, os.path.join(args.restore_dir, args.checkpoint))


# In[28]:


aug = augmentor2d.multi_thread_augmentor()
np.random.seed(0)

df = pd.DataFrame()

for epoch in range(args.nepochs[-1]):
    inds = get_sample_index(classes, args.balance)
    nbatches = len(inds) // args.batch_size
    
    # learning rate scheme
    for k, epoch_th in enumerate(args.nepochs):
        if epoch < epoch_th:
            current_lr = args.lr[k]
            current_trainer = trainers[k]
            break
    
    # get first batch
    next_inds = inds[:args.batch_size]
    aug.start_next_batch_2d(next_inds, imgs, labels)
    batch_x, batch_labels = aug.get_results()
    batch_y = batch_labels[..., [0]]
    batch_mask = batch_labels[..., [1]]
    batch_class = classes[next_inds]
    for ibatch in range(1, len(inds), args.batch_size):
        # start retrieving next batch
        next_inds = inds[ibatch:ibatch+args.batch_size]
        aug.start_next_batch_2d(next_inds, imgs, labels)
        
        # train with current batch
        # first get the prediction
        batch_pred = sess.run(model.pred, {model.X: batch_x, model.mask: batch_mask, model.phase: 0})
        
        # then generate labels according to the prediction
        if epoch > -1:
            batch_z = get_label(np.copy(batch_pred), batch_x, batch_mask, batch_class, args.bias, args.scale)
        else:
            batch_z = batch_y
        
        _, dice_val = sess.run([current_trainer, total_loss], 
            {model.X: batch_x, model.Y: batch_z, model.mask: batch_mask, model.phase: 1, lr: current_lr})
        
        # tensorboard
        add_to_summary(writers['train'], [dice_val], ['dice_batch'], 
                       epoch * nbatches + ibatch // args.batch_size)
        
        # get next batch
        batch_x, batch_labels = aug.get_results()
        batch_y = batch_labels[..., [0]]
        batch_mask = batch_labels[..., [1]]
        batch_class = classes[next_inds]
        
        # print some information
        if (ibatch // args.batch_size + 1) % 10 == 0:
            print ('%d, %d/%d: dice_loss = %g'%                   (epoch, ibatch // args.batch_size, nbatches, dice_val))
            sys.stdout.flush()
    
#     break
    if (epoch + 1) % 5 == 0:
        saver.save(sess, os.path.join(args.output_dir, args.tag, str(epoch)))
    
    # validation and testing
    records = {}
    
    dice_val = evaluate_dice(sess, model, valid_imgs, valid_labels, valid_preds, args)

    print ('%d, %s: dice = %g'%(epoch, dataset, dice_val))

    add_to_record(records, [dice_val], ['dice'], 'train')
    add_to_summary(writers[dataset], [dice_val], ['dice'], epoch * nbatches + ibatch // args.batch_size)
    
    for k in records:
        records[k] = np.mean(records[k])
    df = df.append(records, ignore_index=True)
    df.to_csv(os.path.join(summary_dir, 'logs.csv'), index=False)


# In[ ]:




