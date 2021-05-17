#!/bin/bash

if [[ -z $1 ]]; then
    LOAD_EPOCH=1
else
    LOAD_EPOCH=$1
fi
base_dir=`pwd`

# Where should we save checkpoints and tensorboard events?
JOB_NAME=test_icebert_chkpt${LOAD_EPOCH}_seq512
OUTPUT_DIR=${base_dir}/icebert_model_outputs

# Assumes job name in previous seq128 run, will resume training from epoch 1 by default
CHECKPOINT_BASE_PATH=${OUTPUT_DIR}/saved_models/test_icebert_128
CHECKPOINT_EPOCH_NAME=`basename ${CHECKPOINT_BASE_PATH}/epoch${LOAD_EPOCH}_*`
echo "checkpoint id: $CHECKPOINT_EPOCH_NAME"

mkdir -p $OUTPUT_DIR

deepspeed ${base_dir}/deepspeed_train.py \
--cf ${base_dir}/test_128_512.json \
--max_seq_length 512 \
--output_dir $OUTPUT_DIR \
--print_steps 10 \
--deepspeed \
--deepspeed_transformer_kernel \
--job_name $JOB_NAME \
--deepspeed_config ${base_dir}/test_deepspeed_config.json \
--data_path_prefix /workspace/bert \
--use_nvidia_dataset \
--rewarmup \
--lr_schedule "LN" \
--attention_dropout_checkpoint \
--lr_offset 0.0 \
--load_training_checkpoint ${CHECKPOINT_BASE_PATH} \
--load_checkpoint_id ${CHECKPOINT_EPOCH_NAME} \
&> ${JOB_NAME}.log
