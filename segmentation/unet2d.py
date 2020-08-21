#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from collections import OrderedDict
import argparse
import math


# In[2]:


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def corrupt(x):
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))

def batch_relu(x, phase, scope):
    with tf.variable_scope(scope):
        return tf.cond(phase,  
                lambda: tf.contrib.layers.batch_norm(x, is_training=True, decay=0.9, zero_debias_moving_mean=True,
                                   center=False, updates_collections=None, scope='bnn'),  
                lambda: tf.contrib.layers.batch_norm(x, is_training=False,  decay=0.9, zero_debias_moving_mean=True,
                                   updates_collections=None, center=False, scope='bnn', reuse = True))  


#########################################################################
def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=stddev)
    n_input=shape[2]
    initial= tf.random_uniform(shape,-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input))
    return tf.Variable(initial)

def weight_variable_devonc(shape):
    #return tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    n_input=shape[2]
    initial= tf.random_uniform(shape,-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.00001, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W,keep_prob_):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.dropout(conv_2d, keep_prob_)

def conv2d_stride(x, W,keep_prob_):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d(x, W,stride):
    x_shape = tf.shape(x)
#     x_shape = x.shape
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2,  x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
#     x1_shape = x1.shape
#     x2_shape = x2.shape
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat(axis=3, values=[x1_crop, x2])

# def crop_add(x1,x2):
#     x1_shape = tf.shape(x1)
#     x2_shape = tf.shape(x2)
#     # offsets for the top left corner of the crop
#     offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
#     size = [-1, x2_shape[1], x2_shape[2], -1]
#     x1_crop = tf.slice(x1, offsets, size)
#     return tf.add(x1_crop, x2)
# def crop_add_3rdchannel(x1,x2):
#     x1_shape = tf.shape(x1)
#     x2_shape = tf.shape(x2)
#     # offsets for the top left corner of the crop
#     offsets = [0, 0, 0, 2]
#     size = [x2_shape[0], x2_shape[1], x2_shape[2], x2_shape[3]]
#     x1_crop = tf.slice(x1, offsets, size)
#     return tf.add(x1_crop, x2)


# In[3]:


class unet2d:
    def __init__(self):
        self.input_shape = [256,256,7]
        self.layers = 5
        self.filter_size=3
        self.pool_size=2
        self.features_root=32
        self.keep_prob=1.0
        
        self.dice_smooth = 1
        
        self.n_class=6
        self.fc_nodes = [1024]
        self.fc_drop = [0.25]
        self.up_layers=0
        self.finetune_scope = 'finetune'
    
    def add_to_parser(self, parser):
        for key in vars(self):
            value = vars(self)[key]
            if type(value) == list:
                parser.add_argument('--%s'%key, type=type(value[0]), nargs='+', default=value)
            else:
                parser.add_argument('--%s'%key, type=type(value), default=value)
        return parser
    
    def from_args(self, args):
        for key in vars(args):
            if hasattr(self, key):
                setattr(self, key, getattr(args, key))
    
    def unet(self, X, phase):
        # Build the encoder
        layers = self.layers
        filter_size = self.filter_size
        pool_size = self.pool_size
        features_root = self.features_root
        keep_prob = self.keep_prob
        channels = self.input_shape[-1]
        
        dw_h_convs = OrderedDict()
        in_node = X
        self.down_layer_vars = []
        self.up_layer_vars = []

        # down layers
        for layer in range(0, layers):
            features = 2**layer*features_root
            stddev = np.sqrt(2 / (filter_size**2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features])
            else:
                w1 = weight_variable([filter_size, filter_size, features//2, features])
            w2 = weight_variable([filter_size, filter_size, features, features])
            b1 = bias_variable([features])
            b2 = bias_variable([features])
            
            self.down_layer_vars.append([w1, w2, b1, b2])

            conv1 = conv2d(in_node, w1, keep_prob)
            #tmp_h_conv = tf.nn.relu(conv1 + b1)
            dw_h_convs[layer]=tf.nn.relu(batch_relu(conv1, phase,scope="down1_bn"+str(layer)))

            conv2 = conv2d_stride(dw_h_convs[layer], w2, keep_prob)
            tmp_h_conv= tf.nn.relu(batch_relu(conv2, phase,scope="down2_bn"+str(layer)))
            
            if layer < layers-1:
                #pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node =tmp_h_conv

        in_node = dw_h_convs[layers-1]  
        # up layers
        for layer in range(layers-2, layers-2-self.up_layers, -1):
            features = 2**(layer+1)*features_root

            wd = weight_variable_devonc([pool_size, pool_size, features//2, features])
            bd = bias_variable([features//2])
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)

            w1 = weight_variable([filter_size, filter_size, features, features//2])
            w2 = weight_variable([filter_size, filter_size, features//2, features//2])
            b1 = bias_variable([features//2])
            b2 = bias_variable([features//2])
            
            self.up_layer_vars.append([wd, bd, w1, w2, b1, b2])

            conv1 = conv2d(h_deconv_concat, w1, keep_prob)
            h_conv = tf.nn.relu(batch_relu(conv1, phase,scope="up1_bn"+str(layer)))
            conv2 = conv2d(h_conv, w2, keep_prob)
            in_node = tf.nn.relu(batch_relu(conv2, phase,scope="up2_bn"+str(layer)))
        
        return in_node

    def build_unet(self):
        self.X = tf.placeholder("float", [None] + list(self.input_shape), name='X')
        self.Y = tf.placeholder("float", [None, self.input_shape[0], self.input_shape[1], 1], name='Y')
        self.phase = tf.placeholder(tf.bool, name='phase')
        
        self.up_layers = self.layers - 1
        unet_output = self.unet(self.X, self.phase)
        weight = weight_variable([1, 1, self.features_root, 1])
        bias = bias_variable([1])
        conv = conv2d(unet_output, weight, tf.constant(1.0))
        self.pred_before_sigmoid = conv + bias
        
        self.pred = tf.keras.layers.Activation('sigmoid')(self.pred_before_sigmoid)
        
        self.dice_loss = 1 - (2 * tf.reduce_sum(self.Y * self.pred) + self.dice_smooth) / (tf.reduce_sum(self.Y) + tf.reduce_sum(self.pred) + self.dice_smooth)
    
    def build_unet_mask(self):
        self.X = tf.placeholder("float", [None] + list(self.input_shape), name='X')
        self.Y = tf.placeholder("float", [None, self.input_shape[0], self.input_shape[1], 1], name='Y')
        self.mask = tf.placeholder("float", [None, self.input_shape[0], self.input_shape[1], 1], name='Y')
        self.phase = tf.placeholder(tf.bool, name='phase')
        
        self.up_layers = self.layers - 1
        unet_output = self.unet(self.X, self.phase)
        weight = weight_variable([1, 1, self.features_root, 1])
        bias = bias_variable([1])
        conv = conv2d(unet_output, weight, tf.constant(1.0))
        self.pred_before_sigmoid = conv + bias
        
        self.pred = tf.keras.layers.Activation('sigmoid')(self.pred_before_sigmoid) * self.mask
        
        self.dice_loss = 1 - (2 * tf.reduce_sum(self.Y * self.pred) + self.dice_smooth) / (tf.reduce_sum(self.Y) + tf.reduce_sum(self.pred) + self.dice_smooth)
    
    def build_finetune(self):
        # tf Graph input (only pictures)
        self.X = tf.placeholder("float", [None] + list(self.input_shape), name='X')
        self.Y = tf.placeholder("float", [None, self.n_class], name='Y')
        self.phase = tf.placeholder(tf.bool, name='phase')
        
        unet_output = self.unet(self.X, self.phase)
        
        # the output for classification
        with tf.name_scope(self.finetune_scope):
            self.features = tf.keras.layers.GlobalAveragePooling2D()(unet_output)
            x = self.features
            for fc_node, fc_drop in zip(self.fc_nodes, self.fc_drop):
                x = tf.keras.layers.Dense(fc_node, 'relu')(x)
                x = tf.keras.layers.Dropout(fc_drop)(x, self.phase)
            self.pred_before_sigmoid = tf.keras.layers.Dense(self.n_class)(x)
            self.pred = tf.keras.layers.Activation('sigmoid')(self.pred_before_sigmoid)
        
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.Y, logits = self.pred_before_sigmoid))


# In[4]:


if __name__ == '__main__':
    import subprocess
    subprocess.call(['jupyter', 'nbconvert', '--to', 'script', 'unet2d'])


# In[ ]:




