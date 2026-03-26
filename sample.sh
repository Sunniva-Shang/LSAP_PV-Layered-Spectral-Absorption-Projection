#!/bin/bash
gpus=$8

MODEL_FLAGS="--image_size 128 --num_channels 64 --num_res_blocks 2 --learn_sigma True \
--class_cond True --attention_resolutions 4 --dropout 0.2"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
SAMPLE_FLAGS="--batch_size $2 --num_samples $3"

export OPENAI_LOGDIR=$1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=$gpus python3 scripts/cond_sample.py \
                            --model_path $4 \
                            --outpath $5   \
                            --cond_dir $6   \
                            --sams $7  \
                            $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
