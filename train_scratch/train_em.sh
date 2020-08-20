#!/bin/bash

SCALE_1=(0.25 0.5 0.75 1)
SCALE_2=(1.25 1.5 1.75 2)

for BIAS in -7 -7.5 -8 -8.5 -9 -9.5
do 
    for ((i=0;i<4;i++))
    do 
        python train_em.py --device 1 --bias $BIAS --scale ${SCALE_1[i]} --tag bias_${BIAS}_scale_${SCALE_1[i]} > outputs/bias_${BIAS}_scale_${SCALE_1[i]}.txt &
        python train_em.py --device 2 --bias $BIAS --scale ${SCALE_2[i]} --tag bias_${BIAS}_scale_${SCALE_2[i]} > outputs/bias_${BIAS}_scale_${SCALE_2[i]}.txt
        wait
    done
done