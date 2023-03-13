#!/bin/bash
PYTHON='/home/zitai/anaconda3/envs/pytorch/bin/python'
export CUDA_VISIBLE_DEVICES=1

hostname

# Get unique log file
SAVE_DIR=log/log_train/
SEED=1

# SPECIFY PARAMS
DATASET='svhn'
CLOSE_LOSS='Softmax'
ALPHA=2
LAMBDA=0.0005

AUG_M=18
AUG_N=1

for SPLIT_IDX in 0 1 2 3 4; do

  EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
  EXP_NUM=$((${EXP_NUM}+1))
  echo $EXP_NUM

  ${PYTHON} -m osr  --lr=0.1 \
                    --transform='rand-augment' --rand_aug_m=${AUG_M} --rand_aug_n=${AUG_N} \
                    --close_loss=${CLOSE_LOSS} --openauc=True --alpha=${ALPHA} --lambda=${LAMBDA} \
                    --dataset=${DATASET} --image_size=32 \
                    --batch_size=128 --num_workers=16 --max-epoch=200 \
                    --num_restarts=2 --seed=${SEED} --gpus 0 --feat_dim=128 \
                    --split_idx=${SPLIT_IDX} \
  > ${SAVE_DIR}logfile_${EXP_NUM}.out
done