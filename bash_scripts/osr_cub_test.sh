#!/bin/bash
PYTHON='/home/zitai/anaconda3/envs/pytorch/bin/python'
export CUDA_VISIBLE_DEVICES=5

hostname
# nvidia-smi

# Get unique log file
SAVE_DIR=log/log_test/

# SPECIFY PARAMS
DATASET='cub'
LOSS='OpenAUC-Softmax'
EXP_ID='2023-03-11-11:04:07.677896'
EPOCH=599

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m test_fine_grained --model='timm_resnet50_pretrained' --transform='rand-augment' \
             --loss=${LOSS} --exp_id=${EXP_ID} \
            --dataset=${DATASET} --image_size=448 \
            --batch_size=32 --num_workers=16 --gpus 0 --feat_dim=2048 --max_epoch=${EPOCH} \
> ${SAVE_DIR}logfile_${EXP_NUM}.out