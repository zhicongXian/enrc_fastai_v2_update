import os
import sys
from fastai.basics import *
from fastai.vision import *
# from fastai.callbacks import *
import fastai
from ourUtils.torch_utils.aes.conv_ae import CNNAutoencoderFlexible
from ourUtils.pretrain_utils import random_seed, finetune
import torch
import numpy as np
from fastai.layers import ConvLayer, ResBlock
from fastai.vision.all import *

def pretrain(np_seeds=None, nr_aes = 10,src_path = r"C:\Users\erikc\Documents\Data\enrc_data\enrc_data\nr_objects\tmp\images\train"):
    # nr_aes = 2#10 --TODO
    if np_seeds is None:
        np_seeds = np.random.randint(100000, size=nr_aes)
    else:
        if np_seeds.shape[0] < nr_aes:
            raise ValueError(
                f"passed seeds {np_seeds.shape[0]} are smaller than number of aes {nr_aes}")


    # L2 regularization
    wd = 1e-3
    finetune_iterations =  5000 # 200 #5000 --TODO
    epochs = 72  # approx. 10000 iterations with bs of 64
    max_lr = 1e-2
    tie_weights = False

    # we prefer float32 format instead of float64 and to use of GPU if available
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)

    # src_path = r"C:\Users\zhicong\Documents\Data\enrc_data\enrc_data\nr_objects\tmp\images\train" #os.path.join('data', 'nr_objects/images/train') --TODO
    dropout_rate = 0.3
    slope = 0.1

    def conv_and_res(ni, nf, stride=2): return [ConvLayer(
        ni, nf, stride=stride,  act_cls=partial(nn.LeakyReLU, negative_slope=slope)), nn.Dropout2d(p=dropout_rate),
        ResBlock(1, nf,nf, act_cls=partial(nn.LeakyReLU, negative_slope=slope)) ]

    def conv_trans_and_res(ni, nf, last_layer, stride=2):
        if last_layer:
            return [ConvLayer(ni, nf, ks=2, padding= 0, stride=stride, transpose=True), nn.Dropout2d(p=dropout_rate), ResBlock(1, nf, nf, # ni, nf: number of channels in input and output
                                                                                                             act_cls=partial(nn.LeakyReLU, negative_slope=1.0))]
        else:
            return [ConvLayer(ni, nf, stride=stride, ks=2, padding= 0, transpose=True), nn.Dropout2d(p=dropout_rate), ResBlock(1, nf, nf, act_cls=partial(nn.LeakyReLU, negative_slope=slope))]

    nfs = [128, 256]

    # src = (ImageImageList.from_folder(path=src_path).split_none()
    #        .label_from_func(func=lambda x: x)) # the target is the input
    # stats = src.databunch(
    #     bs=len(src.databunch().label_list.train), device="cpu").batch_stats()


    files = get_image_files(src_path)

    src = DataBlock(
        blocks=(ImageBlock, ImageBlock),
        splitter=IndexSplitter([]),  # Equivalent to split_none(), whether to create validation data
        get_y=lambda x: x,  # Each image is its own target
    )
    bs = len(files)
    dls = src.dataloaders(
        files,
        bs=len(files),
        device=torch.device("cpu"),
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    xb, yb = dls.train.one_batch()

    # Equivalent to v1 batch_stats(): per-channel mean and standard deviation
    stats = [
        xb.mean(dim=(0, 2, 3)),
        xb.std(dim=(0, 2, 3), unbiased=True),
    ] # define the initial normalization

    for hl in [16]:
        embd_sz = hl

        def my_loss(preds, target):
            emb_enc, reconstruction = preds
            return F.mse_loss(reconstruction, target)

        for ae_index in range(nr_aes):
            print("\nStart training ae {} with random seed {}".format(
                ae_index, np_seeds[ae_index]))
            random_seed(np_seeds[ae_index])
            tfms = aug_transforms(
                do_flip=False, max_warp=0.0, max_rotate=0.0, max_zoom=1.)
            # data = src.transform(tfms, tfm_y=True).databunch(
            #     bs=bs).normalize(stats=stats, do_y=True)

            src = DataBlock(
                blocks=(ImageBlock, ImageBlock),
                get_items=get_image_files,
                splitter=IndexSplitter([]),
                get_y=lambda x: x,
                batch_tfms=[
                    *tfms,
                    Normalize.from_stats(*stats),
                ],
            )

            data = src.dataloaders(src_path, bs=bs, num_workers=0, drop_last=False)

            # Training batches
            for xb, yb in data.train:
                print(xb.shape, yb.shape)

            ae_model = CNNAutoencoderFlexible(data=data,
                                              nfs=nfs,
                                              layer_enc=conv_and_res,
                                              embd_sz=embd_sz,
                                              layer_dec=conv_trans_and_res,
                                              first_upscale=8,
                                              tie_weights=tie_weights,
                                              batch_norm_ae=False,
                                              dropout_ae=dropout_rate,
                                              )

            learn = Learner(data, ae_model, loss_func=my_loss, wd=wd)
            print("Base directory:", learn.path)
            print("Model directory:", learn.model_dir)
            learn.fit_one_cycle(epochs, lr_max=max_lr, wd=wd)
            finetune(learn, finetune_iterations, lr=max_lr/2.0)
            learn.save(
                f'ae-model-hl-{embd_sz}-idx-{ae_index}')



if __name__ == "__main__":
    pretrain()
