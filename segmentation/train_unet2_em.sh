# training script for unet1
# this is just a demonstration how to train the network with the sample dataset megseg
# please note that medseg dataset is too small to get a good results. 

#!/bin/bash

INPUT_DIR='/raid/COVID-19/lung_seg_em/output/medseg/npzs'
OUTPUT_DIR='/raid/COVID-19/lung_seg_em/train/unet2'
VALID_PATH='/raid/COVID-19/lung_seg_em/output/medseg/npzs/0.npz'
DEVICE=0
b2='9.0'
k2='0.5'


python train_unet2_em.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --device $DEVICE --valid_path $VALID_PATH --b2 $b2 --k2 $k2