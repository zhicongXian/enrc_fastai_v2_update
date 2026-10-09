import os
import sys
from fastai.basics import *
from fastai.vision import *
from fastai.vision.all import *

# from fastai.callbacks import *
import fastai
# import fastai.utils.collect_env
from ourUtils.torch_utils.aes.conv_ae import CNNAutoencoderFlexible
from ourUtils.pretrain_utils import random_seed, finetune

USE_GPU = True

@torch.no_grad()
def channel_stats(dl):
    """Streaming per-channel statistics for batches shaped [B, C, H, W]."""
    count = 0
    mean = None
    m2 = None  # Sum of squared deviations from the running mean
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    for xb, *_ in dl:
        # Float64 improves numerical accuracy; only one batch is converted.
        x = xb.to(device=device, dtype=torch.float64)

        n = x.shape[0] * x.shape[2] * x.shape[3]
        batch_var, batch_mean = torch.var_mean(
            x, dim=(0, 2, 3), correction=0
        )
        batch_m2 = batch_var * n

        if mean is None:
            mean = batch_mean
            m2 = batch_m2
            count = n
        else:
            total = count + n
            delta = batch_mean - mean

            m2 = m2 + batch_m2 + delta.square() * (count * n / total)
            mean = mean + delta * (n / total)
            count = total

    if count <= 1:
        raise ValueError("At least two pixels per channel are required.")

    # Matches your original std(..., unbiased=True).
    std = (m2 / (count - 1)).clamp_min(0).sqrt()

    # Avoid division by zero for a constant channel.
    std = torch.where(std == 0, torch.ones_like(std), std)

    return [mean.float(), std.float()]

def pretrain(np_seeds=None, nr_aes = 2, src_path = r"C:\Users\erikc\Documents\Data\enrc_data\enrc_data\tmp_cmnist"):
    # nr_aes = 10
    if np_seeds is None:
        np_seeds = np.random.randint(100000, size=nr_aes)
    else:
        if np_seeds.shape[0] < nr_aes:
            raise ValueError(
                f"passed seeds {np_seeds.shape[0]} are smaller than number of aes {nr_aes}")

    bs = 128
    # L2 regularization
    wd = 1e-3
    finetune_iterations = 5000 #  100 #5000 # --TODO
    epochs = 22 # 2 #22  # approx. 10000 iterations with bs of 128 --TODO
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
    # src_path = os.path.join('data', 'cmnist')

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

    # Used to fit the convolution
    def crop_4pxls(x): return x[:, :, 4:-4, 4:-4]

    nfs = [64, 128]

    ############# fastai v1 #############
    # src = (ImageImageList.from_folder(path=src_path, convert_mode='L')
    #        .split_none()
    #        .label_from_func(func=lambda x: x, convert_mode='L'))

    # convert_mode='L' means convert each image to 8-bit grayscale when loading it.
    # - L stands for luminance: one channel representing brightness.
    #####################################


    src = DataBlock(
        blocks=(
            ImageBlock(cls=PILImageBW),  # Input: grayscale
            ImageBlock(cls=PILImageBW),  # Target: grayscale
        ),
        get_items=get_image_files,
        splitter=IndexSplitter([]),  # All images used for training
        get_y=noop,  # Each image is its own target
    )

    # data = src.dataloaders(
    #     src_path,
    #     bs=64,
    #     num_workers=0,
    #     drop_last=False,
    # )

    # stats = src.databunch(
    #     bs=len(src.databunch().label_list.train), device="cpu").batch_stats()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_bs = 64  # Reduce this further if RAM is limited
    dls = src.dataloaders(
        src_path,
        bs=stats_bs,
        device=device,
        # device=torch.device("cpu"),
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    stats = channel_stats(dls.train)


    for hl in [16]:
        embd_sz = hl

        def my_loss(preds, target):
            emb_enc, reconstruction = preds
            return F.mse_loss(reconstruction, target)

        for ae_index in range(nr_aes):
            print("\nStart training ae {} with random seed {}".format(
                ae_index, np_seeds[ae_index]))
            random_seed(np_seeds[ae_index])
            # flip and lightning doesnt make sense here (have no light source and pictures are concatenated)
            ########## fastai v1 ###############
            # tfms = get_transforms(do_flip=False, max_lighting=None)
            #######################################
            tfms = aug_transforms(
                do_flip=False, max_lighting=0.0)

            ########### fastai v2 ##########
            # data = (src.transform(tfms, tfm_y=True)
            #         .databunch(bs=bs)
            #         .normalize(stats=stats, do_y=True))
            ################################


            src = DataBlock(
                blocks=(
                    ImageBlock(cls=PILImageBW),  # Grayscale input
                    ImageBlock(cls=PILImageBW),  # Grayscale target
                ),
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

            ae_model = CNNAutoencoderFlexible(data=data,
                                              nfs=nfs,
                                              layer_enc=conv_and_res,
                                              embd_sz=embd_sz,
                                              layer_dec=conv_trans_and_res,
                                              first_upscale=4,
                                              tie_weights=tie_weights,
                                              batch_norm_ae=False,
                                              dropout_ae=dropout_rate,
                                              output_fn=crop_4pxls,
                                              )

            learn = Learner(data, ae_model, loss_func=my_loss, wd=wd,  path= "./enrc_results/cmnist")
            learn.fit_one_cycle(epochs, lr_max=max_lr, wd=wd)
            finetune(learn, finetune_iterations, lr=max_lr/2.0)
            learn.save(
                f'ae-model-hl-{embd_sz}-idx-{ae_index}')


if __name__ == "__main__":
    pretrain()
