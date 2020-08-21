# scripts for the testing pipeline

#!/bin/bash

INPUT_DIR='/raid/COVID-19/lung_seg_em/output/medseg'
DEVICE=0

# testing of unet1
# python test_unet1.py --input_dir ${INPUT_DIR}/npzs --device $DEVICE
python test_unet2.py --input_dir ${INPUT_DIR}/npzs --device $DEVICE