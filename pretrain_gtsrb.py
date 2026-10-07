import os
import sys
from fastai.basics import *
from fastai.vision import *
# from fastai.callbacks import *
from fastai.vision.all import *

import fastai
# import fastai.utils.collect_env
from ourUtils.torch_utils.aes.conv_ae import CNNAutoencoderFlexible
from fastai.layers import ConvLayer, ResBlock
from ourUtils.pretrain_utils import random_seed, finetune


def pretrain(np_seeds=None, nr_aes = 10, src_path = r"C:\Users\erikc\Documents\Data\enrc_data\enrc_data\gtsrb\tmp\train"):
    nr_aes = 2 # 10 # --TODO
    if np_seeds is None:
        np_seeds = np.random.randint(100000, size=nr_aes)
    else:
        if np_seeds.shape[0] < nr_aes:
            raise ValueError(
                f"passed seeds {np_seeds.shape[0]} are smaller than number of aes {nr_aes}")

    bs = 64
    # L2 regularization
    wd = 1e-3
    finetune_iterations = 2 # 5000 # --TODO
    epochs = 2 # 106  # approx. 10000 iterations with bs of 64
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
    # src_path = os.path.join(
    #     'data', 'gtsrb')

    dropout_rate = 0.1
    slope = 0.1

    def conv_and_res(ni, nf, stride=2): return [ConvLayer(
        ni, nf, stride=stride,  act_cls=partial(nn.LeakyReLU, negative_slope=slope)), nn.Dropout2d(p=dropout_rate),
        ResBlock(1, nf,nf, act_cls=partial(nn.LeakyReLU, negative_slope=slope)) ]

    def conv_trans_and_res(ni, nf, last_layer, stride=2):
        if last_layer:
            return [ConvLayer(ni, nf, ks=2, padding= 0, stride=stride, transpose=True),
                    nn.Dropout2d(p=dropout_rate), ResBlock(1, nf, nf, # ni, nf: number of channels in input and output
                    act_cls=partial(nn.LeakyReLU, negative_slope=1.0))]
        else:
            return [ConvLayer(ni, nf, stride=stride, ks=2, padding= 0, transpose=True),
                    nn.Dropout2d(p=dropout_rate),
                    ResBlock(1, nf, nf, act_cls=partial(nn.LeakyReLU, negative_slope=slope))]

    nfs = [128, 256]
    #### remove below from fastai v1
    # src = (ImageImageList.from_folder(path=src_path, extensions='.png')
    #        .split_none()
    #        .label_from_func(func=lambda x: x))
    #### add above
    src = DataBlock(
        blocks=(ImageBlock, ImageBlock),
        splitter=IndexSplitter([]),  # Equivalent to split_none(), whether to create validation data, IndexSplitter([]) puts every image into training.
        get_y=lambda x: x,  # Each image is its own target
    )
    # High level API to quickly get your data in a DataLoaders

    # stats = src.databunch(
    #     bs=len(src.databunch().label_list.train), device="cpu").batch_stats()
    # It computes the mean and standard deviation of each image channel across the entire training set, using one large batch on the CPU.

    # The corresponding version for fastai v2

    files = get_image_files(src_path)

    dls = src.dataloaders(
        files,
        bs=len(files),  # All images, since you use IndexSplitter([])
        device=torch.device("cpu"),
        num_workers=0,
        drop_last=False,
    )

    x, y = dls.train.one_batch()  # x shape: [N, C, H, W]

    stats = [
        x.mean(dim=(0, 2, 3)),
        x.std(dim=(0, 2, 3)),
    ]

    for hl in [16]:
        embd_sz = hl

        def my_loss(preds, target):
            emb_enc, reconstruction = preds
            return F.mse_loss(reconstruction, target)

        for ae_index in range(nr_aes):
            print("\nStart training ae {} with random seed {}".format(
                ae_index, np_seeds[ae_index]))
            random_seed(np_seeds[ae_index])
            # flip and max_warp doesnt make sense here (flip changes meaning of the traffic sign and max_warp distorts the already distorted images even more)

            ####### fastai v1 ##############
            # tfms = get_transforms(do_flip=False, max_warp=None)
            ################################
            tfms = aug_transforms(
                do_flip=False, max_warp=0.0, max_rotate=0.0, max_zoom=1.)

            ####### fastai v1 ##############
            # data = (src.transform(tfms, tfm_y=True)
            #         .databunch(bs=bs)
            #         .normalize(stats=stats, do_y=True))
            ################################

            # fastai v2:

            src = DataBlock(
                blocks=(ImageBlock, ImageBlock),
                get_items=partial(get_files, extensions=['.png']),
                splitter=IndexSplitter([]),
                get_y=lambda x: x,
                batch_tfms=[ #
                    *tfms,
                    Normalize.from_stats(*stats),
                ],
            )
            # The standard Normalize.from_stats transform is explicitly type-annotated to only target TensorImage
            data = src.dataloaders(src_path, bs=bs, num_workers=0, drop_last=False)

            for xb, yb in data.train:
                print(xb.shape, yb.shape)
                # to check if target also gets normalized:
                print(f"are all tensors equal: {torch.equal(xb, yb)}")


            data = src.dataloaders(src_path, bs=bs, num_workers=0, drop_last=False)

            ae_model = CNNAutoencoderFlexible(data=data,
                                              nfs=nfs,
                                              layer_enc=conv_and_res,
                                              embd_sz=embd_sz,
                                              layer_dec=conv_trans_and_res,
                                              first_upscale=2,
                                              tie_weights=tie_weights,
                                              batch_norm_ae=False,
                                              dropout_ae=dropout_rate,
                                              )

            learn = Learner(data, ae_model, loss_func=my_loss, wd=wd)
            ######## fastai v1 #######
            # learn.fit_one_cycle(epochs, max_lr=max_lr, wd=wd)
            ##########################
            learn.fit_one_cycle(epochs, lr_max = max_lr, wd=wd)


            finetune(learn, finetune_iterations, lr=max_lr/2.0)
            learn.save(
                f'ae-model-hl-{embd_sz}-idx-{ae_index}')


if __name__ == "__main__":
    pretrain()
