#!/bin/bash

if [[ -z $1 ]]; then
    LOAD_EPOCH=1
else
    LOAD_EPOCH=$1
fi
base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=icebert_large_chkpt${LOAD_EPOCH}_seq512
OUTPUT_DIR=${base_dir}/icebert_model_outputs

# Assumes job name in previous seq128 run, will resume training from epoch 18 by default
CHECKPOINT_BASE_PATH=${OUTPUT_DIR}/saved_models/icebert_large_128
CHECKPOINT_EPOCH_NAME=`basename ${CHECKPOINT_BASE_PATH}/epoch${LOAD_EPOCH}_*`
echo "checkpoint id: $CHECKPOINT_EPOCH_NAME"

mkdir -p $OUTPUT_DIR

deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/icebert_large_128_512.json \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 5000 \
--lr_schedule "LN" \
--lr_offset 0.0 \
--rewarmup \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/deepspeed_config_icebert_large.json \
--data_path_prefix /workspace/bert \
--use_nvidia_dataset \
--attention_dropout_checkpoint \
--load_training_checkpoint ${CHECKPOINT_BASE_PATH} \
--load_checkpoint_id ${CHECKPOINT_EPOCH_NAME} \
&> ${JOB_NAME}.log
