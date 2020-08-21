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
parser.add_argument('--input_dir', type=str, help='all the npz files are put here')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--valid', type=str, default=None, help='the npz file for validation')
parser.add_argument('--test', type=str, default=None, help='the npz file for testing')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, nargs='+', default=[0.01, 0.001, 0.0001])
parser.add_argument('--lr_epochs', type=int, nargs='+', default=[50, 100, 200])

# add network parameters
net = unet2d.unet2d()
parser = net.add_to_parser(parser)

# add augmentor parameters
aug = augmentor2d.multi_thread_augmentor()
parser = aug.add_to_parser(parser)

# tensorboard and quick csv view of the results
def add_to_summary(summary_writer, loss_vals, loss_names, global_step):
    summary = tf.Summary()
    for val, name in zip(loss_vals, loss_names):
        summary.value.add(tag = name, simple_value = val)
    summary_writer.add_summary(summary, global_step)

def add_to_record(records, loss_vals, loss_names, phase):
    for val, name in zip(loss_vals, loss_names):
        records['%s_%s'%(name, phase)].append(val)
    return records

if __name__ == '__main__':
    args = parser.parse_args()
    assert(len(args.lr) == len(args.lr_epochs))
    for k in vars(args):
        print (k, '=', vars(args)[k])
    
    # build network
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    tf.reset_default_graph()
    model = unet2d.unet2d()
    model.from_args(args)
    model.build_unet()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    lr = tf.placeholder(tf.float32, name = 'lr')
    with tf.control_dependencies(update_ops):
        trainer = tf.train.AdamOptimizer(lr).minimize(model.dice_loss)

    saver = tf.train.Saver(max_to_keep=args.lr_epochs[-1])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # load training data
    list_filename = glob.glob(os.path.join(args.input_dir, '*.npz'))
    imgs = {'train': [], 'valid': [], 'test': []}
    labels = {'train': [], 'valid': [], 'test': []}
    for filename in list_filename:
        dataset = os.path.basename(filename)[:-len('.npz')]
        f = np.load(filename)
        if dataset == args.valid:
            imgs['valid'].append(f['img'])
            labels['valid'].append(np.where(f['label'] > 0, 1, 0))
        elif dataset == args.test:
            imgs['test'].append(f['img'])
            labels['test'].append(np.where(f['label'] > 0, 1, 0))
        else:
            imgs['train'].append(f['img'])
            labels['train'].append(np.where(f['label'] > 0, 1, 0))

    for k in imgs:
        if len(imgs[k]) > 0:
            imgs[k] = np.concatenate(imgs[k])
            labels[k] = np.concatenate(labels[k])
    
    # tensorboard
    summary_dir = os.path.join(args.output_dir, 'log')
    train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'valid'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'test'), sess.graph)
    
    # the following are the network training code
    aug = augmentor2d.multi_thread_augmentor()
    np.random.seed(0)

    df = pd.DataFrame()

    for epoch in range(args.lr_epochs[-1]):
        inds = np.arange(imgs['train'].shape[0])
        np.random.shuffle(inds)
        nbatches = len(inds) // args.batch_size

        # learning rate scheme
        for lr_val, lr_epoch in zip(args.lr, args.lr_epochs):
            if epoch < lr_epoch:
                current_lr = lr_val
                break

        # record average for each epoch and add to csv for quick view
        records = {}
        for phase in ['train', 'valid', 'test']:
            records['dice_%s'%phase] = []

        # get first batch
        aug.start_next_batch_2d(inds[:args.batch_size], imgs['train'], labels['train'])
        batch_x, batch_y = aug.get_results()
        for ibatch in range(1, len(inds), args.batch_size):
            # start retrieving next batch
            aug.start_next_batch_2d(inds[ibatch:ibatch+args.batch_size], imgs['train'], labels['train'])
            
            # train with current batch
            _, dice_val = sess.run(
                [trainer, model.dice_loss], 
                {model.X: batch_x, model.Y: batch_y, model.phase: 1, lr: current_lr})

            # tensorboard
            add_to_summary(train_writer, [dice_val], ['dice'], epoch * nbatches + ibatch // args.batch_size)

            # csv
            records = add_to_record(records, [dice_val], ['dice'], 'train')

            # get next batch 
            batch_x, batch_y = aug.get_results()

            # print some information
            if (ibatch // args.batch_size + 1) % 10 == 0:
                print ('%d, %d/%d: dice = %g'%(epoch, ibatch // args.batch_size, nbatches, dice_val))

        if (epoch + 1) % 5 == 0:
            saver.save(sess, os.path.join(args.output_dir, str(epoch)))

        # validation and testing
        for dataset in ['valid', 'test']:
            if len(imgs[dataset]) == 0:
                continue
            
            for ibatch in range(0, imgs[dataset].shape[0], args.batch_size):
                batch_x = imgs[dataset][ibatch:ibatch+args.batch_size]
                batch_y = labels[dataset][ibatch:ibatch+args.batch_size]

                dice_val = sess.run(model.dice_loss, {model.X: batch_x, model.Y: batch_y, model.phase: 0})

                # csv
                records = add_to_record(records, [dice_val], ['dice'], dataset)

            records['dice_%s'%dataset] = np.mean(records['dice_%s'%dataset])
            # tensorboard
            if dataset == 'valid':
                add_to_summary(valid_writer, [records['dice_valid']], ['dice'], (epoch + 1) * nbatches)
            else:
                add_to_summary(test_writer, [records['dice_test']], ['dice'], (epoch + 1) * nbatches)

            print ('%d, %s: dice = %g'%(epoch, dataset, records['dice_valid']))

        # save csv
        for k in records:
            records[k] = np.mean(records[k])
        df = df.append(records, ignore_index=True)
        df.to_csv(os.path.join(summary_dir, 'losses.csv'), index=False)