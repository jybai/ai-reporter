#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES

ARCH=$1 # unet, msunet
SPLIT=$2 # 'test'
SRC_DIR=$3 # "${DATA_DIR}/input"
TGT_DIR=$4 # "${DATA_DIR}/output"
CKPT_FILE=$5
ADDITIONAL_FLAGS=$6 # "--load_from_pl --save_results_dir [DIR] --n_saves 10

python eval_unet.py \
  --arch $ARCH --src-dir $SRC_DIR --tgt-dir $TGT_DIR \
  --img-size 1024 --ckpt-file $CKPT_FILE $ADDITIONAL_FLAGS

