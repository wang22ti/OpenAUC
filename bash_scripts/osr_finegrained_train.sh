#!/bin/bash
PYTHON='/home/zitai/anaconda3/envs/pytorch/bin/python'
export CUDA_VISIBLE_DEVICES=1

hostname

# Get unique log file
SAVE_DIR=log/log_train/
SEED=0

# SPECIFY PARAMS
DATASET='cub'
CLOSE_LOSS='Softmax'
ALPHA=2
LAMBDA=0.05

AUG_M=30
AUG_N=2
LABEL_SMOOTHING=0.3


EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m osr --lr=0.001 --model='timm_resnet50_pretrained' --resnet50_pretrain='places_moco' \
                             --transform='rand-augment' \
                            --rand_aug_m=${AUG_M} --rand_aug_n=${AUG_N} \
                            --close_loss=${CLOSE_LOSS} --openauc=True --alpha=${ALPHA} --lambda=${LAMBDA} \
                            --label_smoothing=${LABEL_SMOOTHING} \
                            --dataset=${DATASET} --image_size=448 \
                            --scheduler='cosine_warm_restarts_warmup' --split_train_val='False' --batch_size=32 --num_workers=16 --max-epoch=600 \
                            --num_restarts=2 --seed=${SEED} --gpus 0 --feat_dim=2048 \
> ${SAVE_DIR}logfile_${EXP_NUM}.out