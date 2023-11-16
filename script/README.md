# Image-to-image translation for microscopic images under common input distribution shift
The goal of this project is to examine whether image-to-image translation models for microscopic images of cells (e.g. Unet) are robust under input distribution shift and how to adapt to shifts with semi-supervision.
The input distirbution shift may occur in the following forms:
* different cell staining
* different magnifying factor
* different contrast
* different cell density
* different cell age

We currently support the following strategies for training Unet: Pixel-wise (paired), GAN (unpaired), and ADDA (unpaired).

## Training
Training Unet on unpaired data from scratch is hard, especially on small amount of data.
The pretrained Unet is used to initialize the GAN/ADDA models which impacts the performance significantly.
Pretrained Unet can be trainined on a different input domain mapping to the same output domain.
For ADDA, in addition for initialization the pretrained Unet is furthered used to project the `SRC_A` &rarr; `TGT` for the model to learn `SRC_B` &rarr; `TGT`.

### Pixel-wise Unet (paired input-output)
```
./train_unet.sh [SRC_DIR] [TGT_DIR] [SAVE_CKPT_PATH]
```
### Unet GAN (unpaired input-output)
```
./train_gan.sh [SRC_DIR] [TGT_DIR] [LOG_DIR] [PRETRAINED_UNET_PATH] 
```
### Unet ADDA (unpaired input-output)
```
./train_adda.sh [SRC_A_DIR] [SRC_B_DIR] [TGT_B_DIR] [LOG_DIR] [PRETRAINED_A_UNET_PATH] [KWARGS]
```
`KWARGS` are optinal but should at least specify `--max_epochs [MAX_EPOCHS]`.
### Evaluation
#### pixel-wise Unet
```
./eval_unet_v2.sh [MODEL_ARCH|'unet','msunet'] [DATA_SPLIT|'train','test'] [SRC_A_DIR] [TGT_A_DIR] [LOAD_CKPT_PATH]
```
#### Unet GAN and Unet ADDA
```
./eval_unet_v2.sh [MODEL_ARCH|'unet','msunet'] [DATA_SPLIT|'train','test'] [SRC_A_DIR] [TGT_A_DIR] [LOAD_CKPT_PATH] "--load_from_pl"
```
