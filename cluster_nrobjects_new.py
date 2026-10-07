import os
from ourUtils.torch_utils.aes.conv_ae import CNNAutoencoderFlexible
from fastai.vision.all import *
from fastai.layers import ConvLayer, ResBlock
import torch.utils.data
from torch import nn
import torchvision.transforms as transforms
from ourUtils.torch_utils import clevr_data
from ourUtils.funct_band_aids import setup_directory
import numpy as np
from copy import deepcopy
from pretrain_nrobjects_new import pretrain
from ourUtils.enrc_utils import random_seed, calc_nmi, run_enrc, get_embedding
from fastai.vision.all import *
from fastcore.foundation import L
import argparse

def init_model(embd_sz, nfs, clevr_dir, random_state):
    dropout_rate = 0.3
    slope = 0.1
    tie_weights = False

    def conv_and_res(ni, nf, stride=2): return [ConvLayer(
        ni, nf, stride=stride, act_cls=partial(nn.LeakyReLU, negative_slope=slope)), nn.Dropout2d(p=dropout_rate),
        ResBlock(1, nf, nf, act_cls=partial(nn.LeakyReLU, negative_slope=slope))]

    def conv_trans_and_res(ni, nf, last_layer, stride=2):
        if last_layer:
            return [ConvLayer(ni, nf, ks=2, padding= 0, stride=stride, transpose=True), nn.Dropout2d(p=dropout_rate), ResBlock(1, nf, nf, # ni, nf: number of channels in input and output
                                                                                                             act_cls=partial(nn.LeakyReLU, negative_slope=1.0))]
        else:
            return [ConvLayer(ni, nf, stride=stride, ks=2, padding= 0, transpose=True), nn.Dropout2d(p=dropout_rate), ResBlock(1, nf, nf, act_cls=partial(nn.LeakyReLU, negative_slope=slope))]

    # from fastai.vision import ImageImageList
    # src = (ImageImageList.from_folder(path=clevr_dir)
    #        .split_by_rand_pct(valid_pct=0.1, seed=random_state)
    #        .label_from_func(func=lambda x: x))


    block = DataBlock(
        blocks=(ImageBlock, ImageBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.1, seed=random_state),
        get_y=noop,  # Return the same image path as the target
        batch_tfms=Normalize(),
    )

    src = block.datasets(clevr_dir)

    # Randomly assign approximately 90% to training and 10% to validation. A fixed seed makes the split reproducible for the same ordered file list.

    # bs doesn't matter only shape is important
    data = block.dataloaders(clevr_dir, bs=16, num_workers=0, drop_last=False) # src.databunch(bs=16, drop_last=False)# .normalize() # z-normalization
    ae_model = CNNAutoencoderFlexible(data=data,
                                      nfs=nfs,
                                      layer_enc=conv_and_res,
                                      embd_sz=embd_sz,
                                      layer_dec=conv_trans_and_res,
                                      first_upscale=8,
                                      tie_weights=tie_weights,
                                      batch_norm_ae=False,
                                      dropout_ae=dropout_rate)
    return ae_model


def load_model(model_dir, clevr_dir, nfs, embd_sz, seed):
    ae_model = init_model(embd_sz, nfs, clevr_dir,
                          random_state=seed)
    with torch.serialization.safe_globals([L]):
        ae_dict = torch.load(os.path.join(model_dir), weights_only=False)
        ae_state_dict = ae_dict["model"]
        ae_model.load_state_dict(ae_state_dict, strict=True)
    return ae_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default = r"C:\Users\erikc\Documents\Data\enrc_data\enrc_data\nr_objects\tmp")

    def parse_list(value):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as e:
            raise argparse.ArgumentTypeError(f"Invalid JSON list: {e}")

        if not isinstance(parsed, list):
            raise argparse.ArgumentTypeError("Argument must be a JSON list")

        return parsed

    parser.add_argument('-s', '--seeds', type=parse_list, help='here you can set a list of seeds', default=[1, 2, 3,4,5,6,7,8,9,10])
    # Use like:
    return parser.parse_args()

def main(np_seeds=None):
    args = parse_args()
    clevr_dir = args.dataset_path
    np_seeds = np.asarray(args.seeds)
    nr_aes = 10  # 10 --TODO
    pretrain(np_seeds, nr_aes, clevr_dir)
    CONCAT_LABELS = True

    if np_seeds is None:
        np_seeds = np.random.randint(100000, size=nr_aes)
    else:
        if np_seeds.shape[0] < nr_aes:
            raise ValueError(
                f"passed seeds {np_seeds.shape[0]} are smaller than number of aes {nr_aes}")

    bs = 64
    n_iterations = 20000 #20000 --TODO
    cluster_lr = 1e-2
    pretrain_lr = cluster_lr / 4.0  # initial lr 1e-2/4.0
    # noop returns its argument unchanged, equivalent to lambda x: x
    # Dataset statistics
    # args = parse_args()
    # clevr_dir = args.dataset_path #r"C:\Users\zhicong\Documents\Data\enrc_data\enrc_data\nr_objects\tmp"#os.path.join('data', 'nr_objects') #--TODO
    std = (0.1263, 0.1241, 0.1253)
    mean = (0.4490, 0.4362, 0.4286)

    nr_of_images = 10000 #63# 10000 --TODO

    result_dir = os.path.join("enrc_results", "nr_objects")
    setup_directory(result_dir)

    if CONCAT_LABELS:
        concat_json = clevr_data.generate_scenes(os.path.join(
            clevr_dir, "scenes"), nr_of_images=nr_of_images)
        clevr_data.save_json(concat_json, os.path.join(
            clevr_dir, "scenes", "CLEVR_train_scenes.json"))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean,
                                                         std),
                                    ])
    trainset = clevr_data.ClevrDatasetImagesAndDescriptions(clevr_dir=clevr_dir,
                                                            train=True,
                                                            transform=transform,
                                                            use_cached=False,
                                                            classes=clevr_data.used_classes,
                                                            )

    pt_data, pt_labels = trainset.get_pt_data(N=nr_of_images)
    # Drop size label, because it is only one class "large"
    pt_labels = pt_labels[:, [1, 2, 3]]
    pt_data = pt_data.view((-1, 3, 128, 128))
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(pt_data, torch.from_numpy(pt_labels))), batch_size=bs,
                                             shuffle=False,
                                             drop_last=False,
                                             pin_memory=True,
                                             num_workers=0)

    np.savetxt(os.path.join(result_dir, "labels.csv"),
               pt_labels.astype(np.int64), delimiter=";")

    nfs = [128, 256]
    for hl in [16]:
        embd_sz = hl
        for try_i in range(2):
            for ae_index in range(nr_aes):
                print("\nStart training ae {} with random seed {}".format(
                    ae_index, np_seeds[ae_index]))

                np_seeds[ae_index] += try_i
                random_seed(np_seeds[ae_index])
                model_name = f"ae-model-hl-{embd_sz}-idx-{ae_index}.pth"

                trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(pt_data, torch.from_numpy(pt_labels))),
                                                          batch_size=bs,
                                                          shuffle=True,
                                                          drop_last=False, num_workers=0, pin_memory=True)

                # model_dir = os.path.join(
                #     clevr_dir, "images", "train", "models", model_name)
                model_dir = os.path.join("models", model_name)#"./models"

                ae_model = load_model(model_dir, clevr_dir,
                                      nfs, embd_sz, seed=np_seeds[ae_index])

                ae_model.cuda()
                np_emb = get_embedding(ae_model, testloader)
                np.savetxt(os.path.join(result_dir, model_name.split(
                    ".")[0]+".csv"), np_emb, delimiter=";")

                ae_model, cluster_layer = run_enrc(ae_model=ae_model,
                                                   pt_data=pt_data,
                                                   pt_labels=pt_labels,
                                                   pretrain_lr=pretrain_lr,
                                                   cluster_lr=cluster_lr,
                                                   n_iterations=n_iterations,
                                                   trainloader=trainloader,
                                                   random_state=np_seeds[ae_index])

                # Save Results
                res_df, pred_labels = calc_nmi(
                    ae_model, cluster_layer, testloader, pt_labels, classes=["color", "material", "shape"])
                pred_path = os.path.join(
                    result_dir, model_name.split(".")[0]+"_nmis.csv")
                res_df.to_csv(pred_path, sep=";", index=False) # it already has the function saving
                np.savetxt(os.path.join(result_dir, model_name.split(".")[0]+".labels"),
                           pred_labels.astype(np.int64), delimiter=";")

                del ae_model
                del cluster_layer
                del trainloader
                torch.cuda.empty_cache()


if __name__ == "__main__":
    nr_aes = 10
    # --TODO set seeds
    np_seeds = np.random.randint(100000, size=nr_aes)

    main(np_seeds)
