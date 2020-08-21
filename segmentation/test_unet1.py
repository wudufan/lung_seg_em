from __future__ import print_function
import tensorflow as tf
import numpy as np
import glob
import os
import sys
import argparse
import unet2d

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--device', type=str)

parser.add_argument('--check_dir', type=str, default='../weights/unet1')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=199)

# add network parameters
net = unet2d.unet2d()
parser = net.add_to_parser(parser)

def prediction(file_tag, input_dir, sess, model, batch_size):
    '''
    params:
    @file_tag: short filename excluding dir or postfix
    @input_dir: input directory of the npz, output_dir will be input_dir/unet1
    '''
    
    filename = os.path.join(input_dir, file_tag+'.npz')
    output_dir = os.path.join(input_dir, 'unet1')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # load training data
    imgs = np.load(filename)['img']
    
    # network prediction
    preds = []
    for ibatch in range(0, imgs.shape[0], batch_size):
        print ('%d/%d'%(ibatch, imgs.shape[0]), end=',')
        sys.stdout.flush()
        batch_x = imgs[ibatch:ibatch + args.batch_size]
        pred = sess.run(model.pred, {model.X: batch_x, model.phase: 0})

        preds.append(pred)
    preds = np.concatenate(preds)
    
    # save only the prediction to save some space. During the development everything was saved together
    np.save(os.path.join(output_dir, file_tag), preds)


if __name__ == '__main__':
    args = parser.parse_args()
    for k in vars(args):
        print (k, '=', vars(args)[k])
        
    # build network
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    tf.reset_default_graph()
    model = unet2d.unet2d()
    model.from_args(args)
    model.build_unet()

    loader = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    try:
        loader.restore(sess, os.path.join(args.check_dir, str(args.epoch)))
    except Exception as e:
        print ('Failed to restore from checkpoint:', e)
    
    # prediction
    filetags = [os.path.basename(f)[:-4] for f in glob.glob(os.path.join(args.input_dir, '*.npz'))]
    for filetag in filetags:
        print ('Predicting:', filetag)
        prediction(filetag, args.input_dir, sess, model, args.batch_size)