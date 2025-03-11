#!/bin/bash

for ((i=1;i<6;i+=1))
do 
    # run OTDF on morph task
    CUDA_VISIBLE_DEVICES=0 python train_otdf.py --env halfcheetah-morph --policy OTDF --srctype medium --tartype medium --weight --reg_weight 0.5 --seed $i &

    # run OTDF on gravity task
    CUDA_VISIBLE_DEVICES=0 python train_otdf.py --env halfcheetah-gravity --policy OTDF --srctype medium --tartype medium --weight --reg_weight 0.5 --seed $i &

    # run OTDF on kinematic task
    CUDA_VISIBLE_DEVICES=0 python train_otdf.py --env halfcheetah --policy OTDF --srctype medium --tartype medium --weight --reg_weight 0.1 --seed $i &
done
