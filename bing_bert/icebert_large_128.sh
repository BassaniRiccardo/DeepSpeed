#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=icebert_large_128
OUTPUT_DIR=${base_dir}/icebert_model_outputs

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/icebert_large_128_512.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 5000 \
--lr_schedule "LN" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_config_icebert_large.json \
--data_path_prefix /workspace/bert \
--use_nvidia_dataset \
&> ${JOB_NAME}.log
