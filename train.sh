#!/bin/bash

export OPENAI_LOGDIR="./checkpoint/vein_sty_concat_sigma_cosine_vpem"
export NCCL_IB_DISABLE=1


MODEL_FLAGS="--image_size 128 --num_channels 64 --num_res_blocks 2 --learn_sigma True \
--class_cond True --attention_resolutions 4 --dropout 0.2"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 2e-4 --batch_size 32 --save_interval 10000"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpiexec -n 8 --allow-run-as-root python3 scripts/image_train.py \
       --data_dir traindata/vein_cycle_vpem \
       $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

