#!/usr/bin/env python
# coding: utf-8

# In[6]:


import threading
import numpy as np
import tensorflow as tf


# In[7]:


def allocate_slices_to_threads(i_start, i_end, n_process):
    n_slices = i_end - i_start
    if n_slices < n_process:
        n_process = n_slices
    njobs = np.zeros((n_slices // n_process + 1) * n_process, int)
    njobs[:n_slices] = 1
    
    njobs = njobs.reshape([-1, n_process]).sum(0)
    
    job_ends = np.cumsum(njobs) + i_start
    job_starts = job_ends - njobs
    
    return np.array([job_starts, job_ends]).T


# In[1]:


def augment_2d(islices, src_img, src_seg, results_img, results_seg, generator):
        '''
        generate augmented 2d training dataset
        '''
        for islice in islices:
            transform = generator.get_random_transform(src_img[islice, ...].shape)
            results_img[islice, ...] = generator.apply_transform(src_img[islice, ...], transform)
            results_seg[islice, ...] = generator.apply_transform(src_seg[islice, ...], transform)


# In[2]:


class multi_thread_augmentor:
    def __init__(self):
        self.threads = []
        self.rotation_range = 90
        self.width_shift_range = 0.1
        self.height_shift_range = 0.1
        self.zoom_range = [0.9, 1.1]
        self.horizontal_flip = 1
        self.vertical_flip = 1
        self.n_process = 8
        
    def add_to_parser(self, parser):
        for key in vars(self):
            if key in ['threads']:
                continue
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
    
    def start_next_batch_2d(self, islices, original_imgs, original_segs):
        assert (len(self.threads) == 0)
        
        # allocate slices to different threads
        src_img = original_imgs[islices]
        src_seg = original_segs[islices]
        
        job_slices = allocate_slices_to_threads(0, len(src_img), self.n_process)
                
        # create buffer to hold results
        self.aug_img = np.zeros_like(src_img)
        self.aug_seg = np.zeros_like(src_seg)
        
        generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range = self.rotation_range, 
            width_shift_range = self.width_shift_range, 
            height_shift_range = self.height_shift_range, 
            zoom_range = self.zoom_range,
            fill_mode = 'constant', 
            cval = 0, 
            horizontal_flip = self.horizontal_flip, 
            vertical_flip = self.vertical_flip)
        
        self.threads = []
        for i in range(len(job_slices)):
            islices = np.arange(job_slices[i][0], job_slices[i][1])
            thread = threading.Thread(target = augment_2d, args = (islices, src_img, src_seg, self.aug_img, self.aug_seg, generator))
            self.threads.append(thread)
            thread.start()
    
    def get_results(self):
        for thread in self.threads:
            thread.join()
        
        self.threads = []
        return np.copy(self.aug_img), np.where(self.aug_seg > 0.5, 1, 0)


# In[3]:


if __name__ == '__main__':
    import subprocess
    subprocess.call(['jupyter', 'nbconvert', '--to', 'script', 'augmentor2d'])


# In[ ]:




