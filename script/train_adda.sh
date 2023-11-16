#/bin/bash
export CUDA_VISIBLE_DEVICES
set -e

DIR_A_SRC=$1 # "/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC_A}/input"
DIR_B_SRC=$2 # "/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC_B}/input"
DIR_B_TGT=$3 # "/home/andrewbai/data/cell_img2img/${TGT}_MSC_20x_${SRC_B}/output"
LOG_DIR=$4 # "./adda_logs/${SRC_B}_${TGT}"
UNET_PATH=$5 # "./checkpoints/baseline_${TGT}_${SRC_A}.ckpt"
KWARGS=$6 # --max_epochs 200 --save_every_n_epochs 20, --max_B_size 20, --name ntrn-100

mkdir -p $LOG_DIR

python train_adda.py --accelerator 'gpu' --auto_select_gpus 'True' --devices 1 \
  --data_dir_A_src $DIR_A_SRC --data_dir_B_src $DIR_B_SRC --data_dir_B_tgt $DIR_B_TGT --pretrained_unet_path $UNET_PATH \
  --save_dir $LOG_DIR $KWARGS
