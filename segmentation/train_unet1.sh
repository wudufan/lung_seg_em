# training script for unet1
# this is just a demonstration how to train the network with the sample dataset megseg

#!/bin/bash

INPUT_DIR='/raid/COVID-19/lung_seg_em/output/medseg/npzs'
OUTPUT_DIR='/raid/COVID-19/lung_seg_em/train/unet1'
DEVICE=0
VALID='NONE'
TEST='NONE'

python train_unet1.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --device $DEVICE --valid $VALID --test $TEST