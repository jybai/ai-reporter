import os
import argparse
import torch
from torchmetrics import PearsonCorrCoef
# from scipy.stats import pearsonr
import numpy as np
from tqdm.auto import tqdm, trange
import warnings
from imageio import imwrite
from tifffile import imwrite as tif_imwrite

from networks import define_G
# from BNNBench.backbones.unet import define_G
# from BNNBench.trainer.ensemble_trainer import test_epoch
from BNNBench.data.paired_data import get_loader_with_dir

from utils import get_constant_dim_mask, remove_prefix
from msu_net import MSU_Net

def process_imgs(imgs):
    # transform from [-1, 1] to [0, 255]
    imgs = (((imgs * 0.5) + 0.5) * 255).clip(0, 255).astype(np.uint8)

    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, 1)
    elif imgs.ndim == 4:
        assert imgs.shape[1] in [1, 3]
    else:
        raise NotImplementedError
    return imgs

def parse_arguments():
    parser = argparse.ArgumentParser("Launch the ensemble evaluater")
    parser.add_argument("--arch", type=str, choices=['unet', 'msunet'])
    parser.add_argument("--split", type=str, default='test', choices=['train', 'test'])
    parser.add_argument(
        "--src-dir",
        type=str,
        required=True,
        help="Path to the testing source directory",
    )
    parser.add_argument(
        "--tgt-dir",
        type=str,
        required=True,
        help="Path to the testing target directory",
    )
    parser.add_argument("--ckpt-file", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--img-size", type=int, default=1024, help="Size of images in pixels")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load_from_pl", action="store_true")

    parser.add_argument("--save_results_dir", type=str, default=None)
    parser.add_argument("--n_saves", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.src_dir = os.path.join(args.src_dir, args.split)
    args.tgt_dir = os.path.join(args.tgt_dir, args.split)
    test_loader, fnames = get_loader_with_dir(
        args.src_dir, args.tgt_dir, args.img_size, args.batch_size, is_train=False
    )

    src_batch, tgt_batch = iter(test_loader).__next__()
    in_nc = src_batch.size(1)
    out_nc = tgt_batch.size(1)

    if args.arch == 'unet':
        model = define_G(in_nc, out_nc, 64, "unet_256", norm="batch", use_dropout=False)
        state_dict = torch.load(args.ckpt_file)
        if args.load_from_pl:
            state_dict = state_dict["state_dict"]
            prefix = "G."
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        else:
            state_dict = {remove_prefix(k, 'module.'): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
    elif args.arch == 'msunet':
        # model = MSU_Net(in_nc, out_nc)
        model = define_G(in_nc, out_nc, 64, "msunet_256", norm="batch", use_dropout=False)
        state_dict = torch.load(args.ckpt_file)
        if args.load_from_pl:
            state_dict = state_dict["state_dict"]
            prefix = "model."
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError
    model.eval().to(device)

    # test_epoch(test_loader, model)

    # TODO: hack!!!!
    # mask = np.array([False, False, True])
    # warnings.warn(f"Currently using fixed mask channel mask {mask}.")
    mask = get_constant_dim_mask(tgt_batch[0].detach().cpu().numpy())
    mask = torch.from_numpy(mask).to(device)

    corrs = []
    pearson = PearsonCorrCoef().to(device)
    all_pred_imgs, all_tgt_imgs = [], []
    for src_imgs, tgt_imgs in tqdm(test_loader, leave=False):

        src_imgs = src_imgs.to(device)
        tgt_imgs = tgt_imgs.to(device)
        with torch.no_grad():
            pred_imgs = model(src_imgs)

        tgt_imgs = tgt_imgs[:, mask]
        pred_imgs = pred_imgs[:, mask]

        for pred_img, tgt_img in zip(pred_imgs, tgt_imgs):
            # c, pvalue = pearsonr(pred_img.reshape(-1), tgt_img.reshape(-1))
            c = pearson(pred_img.view(-1), tgt_img.view(-1)).item()
            corrs.append(c)

        all_pred_imgs.append(pred_imgs.detach().cpu().numpy())
        all_tgt_imgs.append(tgt_imgs.detach().cpu().numpy())
    print("===> Testing Corr:", np.mean(corrs))

    if args.save_results_dir is not None:
        os.makedirs(args.save_results_dir, exist_ok=True)

        all_pred_imgs = np.concatenate(all_pred_imgs, axis=0)
        all_tgt_imgs = np.concatenate(all_tgt_imgs, axis=0)

        assert len(all_tgt_imgs) == len(fnames)

        all_pred_imgs = process_imgs(all_pred_imgs)
        all_tgt_imgs = process_imgs(all_tgt_imgs)

        for ii, (pred_img, tgt_img, fname) in enumerate(zip(all_pred_imgs, all_tgt_imgs, tqdm(fnames, leave=False))):
            pred_path = os.path.join(args.save_results_dir, f"pred_{fname}")
            tgt_path = os.path.join(args.save_results_dir, f"tgt_{fname}")

            if ('.tif' in fname) or (".tiff" in fname):
                tif_imwrite(pred_path, pred_img)
                tif_imwrite(tgt_path, tgt_img)
            else:
                imwrite(pred_path, pred_img)
                imwrite(tgt_path, tgt_img)

            if (args.n_saves is not None) and (ii + 1 >= args.n_saves):
                break
        print(f"Results saved in {args.save_results_dir}")

if __name__ == '__main__':
    main()

