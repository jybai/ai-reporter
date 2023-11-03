 #/bin/bash
export CUDA_VISIBLE_DEVICES
set -e

SRC_DIR=$1 # "/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC}/input"
TGT_DIR=$2 # "/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC}/output"
LOG_DIR=$3 # "./gan_logs/${SRC}_${TGT}"
KWARGS=$4 # --pretrained_unet_path '' --max_epochs 100  --name ntrn-100

BSIZE=8

mkdir -p $LOG_DIR

python train_msunet.py --accelerator 'gpu' \
  --auto_select_gpus 'True' --strategy 'ddp_find_unused_parameters_false'\
  --data_src_dir $SRC_DIR --data_tgt_dir $TGT_DIR \
  --save_dir $LOG_DIR --bsize $BSIZE $KWARGS

