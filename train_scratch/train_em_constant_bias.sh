#!/bin/bash

BIAS_1=(0.1 0.2 0.3 0.4 0.5)
BIAS_2=(0.6 0.7 0.8 0.9 1.0)
SCALE=-1

for ((i=0;i<5;i++))
do 
    python train_em.py --device 1 --bias ${BIAS_1[i]} --scale ${SCALE} --tag bias_${BIAS_1[i]}_scale_${SCALE} > outputs/bias_${BIAS_1[i]}_scale_${SCALE}.txt &
    python train_em.py --device 2 --bias ${BIAS_2[i]} --scale ${SCALE} --tag bias_${BIAS_2[i]}_scale_${SCALE} > outputs/bias_${BIAS_2[i]}_scale_${SCALE}.txt
    wait
done