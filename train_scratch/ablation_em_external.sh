#!/bin/bash

SCALE_1=(0.25 0.5 0.75 1)
SCALE_2=(1.25 1.5 1.75 2)
BASEDIR=/raid/COVID-19/CT-severity/results/Iran-2020-04-01-with-annotation/unet2d_256x256x7_mask/em/

for BIAS in -7 -7.5 -8 -8.5 -9 -9.5
do 
    for ((i=0;i<4;i++))
    do 
        python ablation_em_external.py --device 1 --restore_dir ${BASEDIR}bias_${BIAS}_scale_${SCALE_1[i]}/ &
        python ablation_em_external.py --device 2 --restore_dir ${BASEDIR}bias_${BIAS}_scale_${SCALE_2[i]}/
        wait
    done
done


BIAS_1=(0.1 0.2 0.3 0.4 0.5)
BIAS_2=(0.6 0.7 0.8 0.9 1.0)
SCALE=-1
BASEDIR=/raid/COVID-19/CT-severity/results/Iran-2020-04-01-with-annotation/unet2d_256x256x7_mask/em/

for ((i=0;i<5;i++))
do 
    python ablation_em_external.py --device 1 --restore_dir ${BASEDIR}bias_${BIAS_1[i]}_scale_${SCALE}/ &
    python ablation_em_external.py --device 2 --restore_dir ${BASEDIR}bias_${BIAS_2[i]}_scale_${SCALE}/
    wait
done