from __future__ import print_function
import tensorflow as tf
import numpy as np
import glob
import os
import sys
import argparse
import unet2d
import augmentor2d
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str)
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--valid_path', type=str, default=None, help='Path to the dice validation npz, e.g. medseg')

parser.add_argument('--b2', type=float)
parser.add_argument('--k2', type=float)

parser.add_argument('--checkpoint', type=str, default='../weights/unet1/199')

parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--balance', type=int, default=1, help='balance between consolidation images and ggo images')

net = unet2d.unet2d()
parser = net.add_to_parser(parser)

aug = augmentor2d.multi_thread_augmentor()
parser = aug.add_to_parser(parser)

def load_data(input_dir):
    list_filename = glob.glob(os.path.join(input_dir, '*.npz'))
    imgs = []
    labels = []
    annotations = []
    for filename in list_filename:
        f = np.load(filename)

        imgs.append(f['img'])
        labels.append(np.where(f['label'] > 0, 1, 0))
        annotations.append(f['annotation'])

    imgs = np.concatenate(imgs)
    labels = np.concatenate(labels)
    annotations = np.concatenate(annotations)
    
    return imgs, labels, annotations

def clean_data(imgs, labels, annotations):
    '''
    Exclude the layers without any infection region;
    Exclude the layers where there is infection but no annotation (GGO=0, consolidation=0). This is not likely to happen.
    '''
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
    
    return imgs[inds], labels[inds], annotations[inds], inds

def extract_class_info(annotations):
    '''
    get the classes (0 = normal, 1 = ggo, 2 = consolidation) for each slice, and retrieve the indices for balanced batch
    '''
    classes = np.zeros(len(annotations), np.float32)
    # air
    inds = np.where(annotations.sum(1) == 0)[0]
    classes[inds] = 0
    # GGO
    inds_ggo = np.where(annotations[:, 0] == 1)[0]
    classes[inds_ggo] = 1
    # consolidation    
    inds_cons = np.where(annotations[:, 1] == 1)[0]
    classes[inds_cons] = 2
    
    return classes, inds_ggo, inds_cons

# tensorboard
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

def get_sample_index(classes, balance = True):
    if not balance:
        inds = np.arange(len(classes))
        np.random.shuffle(inds)
        return inds
    
    unique_class = np.unique(classes)
    cnt_per_class = [np.count_nonzero(classes == s) for s in unique_class]
    target_cnt_per_class = max(cnt_per_class)
    sample_inds = []
    for iclass in unique_class:
        base_inds = list(np.where(classes == iclass)[0])
        n_rep = int(np.ceil(target_cnt_per_class / float(len(base_inds))))
        inds = []
        for i in range(n_rep):
            np.random.shuffle(base_inds)
            inds += base_inds
        sample_inds.append(inds[:target_cnt_per_class])
    
    return np.array(sample_inds).T.flatten()

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

def evaluate_dice(sess, model, imgs, labels, masks, batch_size):
    '''
    dice of the consolidation on the validation dataset
    '''
    preds = []
    
    for ibatch in range(0, imgs.shape[0], batch_size):
        batch_x = imgs[ibatch:ibatch+batch_size]
        batch_mask = masks[ibatch:ibatch+batch_size]
        preds.append(sess.run(model.pred, {model.X: batch_x, model.mask: batch_mask, model.phase: 0}))
    preds = np.concatenate(preds)
    preds_con = np.where(preds > 0.5, 1, 0)
    labels_con = np.where(labels == 2, 1, 0)
    
    dice_val = 2 * np.sum(preds_con * labels_con, dtype = np.float32) / (np.sum(preds_con) + np.sum(labels_con) + 1e-4)
    
    return dice_val

if __name__ == '__main__':
    args = parser.parse_args()
    # development interface
    args.bias = -args.b2
    args.scale = args.k2
    for k in vars(args):
        print (k, '=', vars(args)[k])
    
    # build network
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    tf.reset_default_graph()
    model = unet2d.unet2d()
    model.from_args(args)
    model.build_unet_mask()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        trainer = optimizer = tf.train.AdamOptimizer(args.lr).minimize(model.dice_loss)

    saver = tf.train.Saver(max_to_keep=args.nepochs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # restore
    loader = tf.train.Saver()
    loader.restore(sess, args.checkpoint)
    
    # load training data with some cleaning to extract most relevant samples
    imgs, labels, annotations = load_data(args.input_dir)
    imgs, labels, annotations, inds = clean_data(imgs, labels, annotations)
    classes, inds_ggo, inds_cons = extract_class_info(annotations)
    
    # load validation data
    try:
        valid_data = dict(np.load(args.valid_path))
        valid_data['pred'] = np.load(os.path.join(os.path.dirname(args.valid_path), 'unet1', os.path.basename(args.valid_path)[:-4]+'.npy'))
        valid_data['pred'] = np.where(valid_data['pred'] > 0.5, 1, 0)
    except Exception as e:
        print (e)
        valid_data = None
    
    # tensorboard
    summary_dir = os.path.join(args.output_dir, 'log')
    datasets = ['train', 'valid']
    writers = {}
    for dataset in datasets:
        writers[dataset] = tf.summary.FileWriter(os.path.join(summary_dir, dataset), sess.graph)
    
    # augmentation
    aug = augmentor2d.multi_thread_augmentor()
    np.random.seed(0)

    df = pd.DataFrame()

    # This is the training code
    for epoch in range(args.nepochs):
        inds = get_sample_index(classes, args.balance)
        nbatches = len(inds) // args.batch_size

        # get first batch
        next_inds = inds[:args.batch_size]
        aug.start_next_batch_2d(next_inds, imgs, labels)
        batch_x, batch_mask = aug.get_results()
        batch_class = classes[next_inds]
        for ibatch in range(1, len(inds), args.batch_size):
            # start retrieving next batch
            next_inds = inds[ibatch:ibatch+args.batch_size]
            aug.start_next_batch_2d(next_inds, imgs, labels)

            # train with current batch
            # first get the prediction
            batch_pred = sess.run(model.pred, {model.X: batch_x, model.mask: batch_mask, model.phase: 0})

            # then generate labels according to the prediction
            batch_y = get_label(np.copy(batch_pred), batch_x, batch_mask, batch_class, args.bias, args.scale)

            _, dice_val = sess.run([trainer, model.dice_loss], 
                {model.X: batch_x, model.Y: batch_y, model.mask: batch_mask, model.phase: 1})

            # tensorboard
            add_to_summary(writers['train'], [dice_val], ['dice_batch'], 
                           epoch * nbatches + ibatch // args.batch_size)

            # get next batch
            batch_x, batch_mask = aug.get_results()
            batch_class = classes[next_inds]

            # print some information
            if (ibatch // args.batch_size + 1) % 10 == 0:
                print ('%d, %d/%d: dice_loss = %g'%\
                       (epoch, ibatch // args.batch_size, nbatches, dice_val))
                sys.stdout.flush()

        if (epoch + 1) % 5 == 0:
            saver.save(sess, os.path.join(args.output_dir, str(epoch)))

        # validation and testing
        if valid_data is not None:
            records = {}
            dice_val = evaluate_dice(sess, model, valid_data['img'], valid_data['label'], valid_data['pred'], args.batch_size)

            print ('%d, %s: dice = %g'%(epoch, dataset, dice_val))

            add_to_record(records, [dice_val], ['dice'], 'valid')
            add_to_summary(writers['valid'], [dice_val], ['dice'], epoch * nbatches + ibatch // args.batch_size)

            for k in records:
                records[k] = np.mean(records[k])
            
            df = df.append(records, ignore_index=True)
            df.to_csv(os.path.join(summary_dir, 'logs.csv'), index=False)