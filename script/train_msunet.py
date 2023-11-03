from argparse import ArgumentParser
import numpy as np

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import LitAlignedDM
from model import LitMSUnetV2 as LitModel
# from model import LitMSUnet as LitModel

def parse_arguments():
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--data_src_dir", type=str)
    parser.add_argument("--data_tgt_dir", type=str)
    parser.add_argument("--bsize", type=int, default=8)
    parser.add_argument("--out_imsize", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every_n_epochs", type=int, default=None)

    parser.add_argument("--save_dir", type=str, default="./lightning_logs")
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--version", type=str, default=None)

    parser = LitModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser) # max_epochs

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.max_epochs is None:
        args.max_epochs = 100

    model = LitModel(**vars(args))

    dm_train = LitAlignedDM(src_dir=args.data_src_dir, tgt_dir=args.data_tgt_dir, 
                            out_imsize=args.out_imsize, bsize=args.bsize, 
                            num_workers=args.num_workers)
    dl_train = dm_train.train_dataloader()

    dm_test = LitAlignedDM(src_dir=args.data_src_dir, tgt_dir=args.data_tgt_dir,
                           out_imsize=args.out_imsize, bsize=1, 
                           num_workers=args.num_workers)
    dl_test = dm_test.test_dataloader()

    if args.save_every_n_epochs is None:
        checkpoint_callback = ModelCheckpoint(monitor='pearson_val', mode='max', save_last=True)
    else:
        checkpoint_callback = ModelCheckpoint(
                every_n_epochs=args.save_every_n_epochs, save_top_k=-1, save_last=True)
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    callbacks = [checkpoint_callback, lr_monitor_callback]

    logger = TensorBoardLogger(save_dir=args.save_dir, name=args.name, version=args.version)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, 
                                            log_every_n_steps=min(20, len(dl_train)))
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_test)

if __name__ == '__main__':
    main()

