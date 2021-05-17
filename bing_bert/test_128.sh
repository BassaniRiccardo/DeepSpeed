#!/bin/bash

base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=test_icebert_128
OUTPUT_DIR=${base_dir}/icebert_model_outputs

mkdir -p $OUTPUT_DIR

NCCL_TREE_THRESHOLD=0 deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/test_128_512.json \
--max_seq_length 128 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 10 \
--lr_schedule "LN" \
--lr_offset 10e-4 \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/test_deepspeed_config.json \
--data_path_prefix /workspace/bert \
--use_nvidia_dataset \
&> ${JOB_NAME}.log
