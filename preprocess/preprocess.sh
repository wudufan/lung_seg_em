#!/bin/bash

# directories
INPUT_DIR='/raid/COVID-19/lung_seg_em/data/medseg'
PROCESS_DIR='/raid/COVID-19/lung_seg_em/processed/medseg'
OUTPUT_DIR='/raid/COVID-19/lung_seg_em/output/medseg'

# step 1: lung mask generation:

# Generate the lung mask using https://github.com/JoHof/lungmask
# Run pip install git+https://github.com/JoHof/lungmask first to install the package

LUNG_IN=${INPUT_DIR}/ct
LUNG_OUT=${INPUT_DIR}/lung_mask

mkdir -p $LUNG_OUT

echo 'Generating lung masks'
for src in "$LUNG_IN"/*
do
    dst=${LUNG_OUT}/$(basename $src)
    echo $src
    echo $dst
    
    lungmask $src $dst --modelname R231CovidWeb
    
    break
done

# step 2: resample to 512x512x5mm
echo 'Resampling ct to 512x512x5mm'
python3 resample_medseg.py --input_dir ${INPUT_DIR}/ct --output_dir ${PROCESS_DIR}/512x512x5mm/ct
echo 'Resampling label to 512x512x5mm'
python3 resample_medseg.py --input_dir ${INPUT_DIR}/label --output_dir ${PROCESS_DIR}/512x512x5mm/label --interpolation nearest
echo 'Resampling lung masks to 512x512x5mm'
python3 resample_medseg.py --input_dir ${INPUT_DIR}/lung_mask --output_dir ${PROCESS_DIR}/512x512x5mm/lung_mask --interpolation nearest

# step 3: generate the training npz files
echo 'Generating training npz files'
python3 get_training_npz.py --input_dir ${PROCESS_DIR}/512x512x5mm --output_dir $OUTPUT_DIR