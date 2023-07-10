import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# from efficientnet_pytorch import EfficientNet
from .shadow_dataset import TextureDataset, toTensor, Normalize


def dataset_init(args):
    # Dataset list
    All_test_path = 'data/dataset/val_list.txt'

    img_root = 'data/dataset/img'
    eval_root = 'data/dataset/results'

    val_dataset = TextureDataset(All_test_path, phase='val', image_size=args.img_size,
                                 transform=True,
                                 multi_path=args.multi_path,
                                 # dataset=args.dataset,
                                 dataset='All',
                                 dataset_path=img_root,
                                 detections_path=eval_root,
                                 eval_to_list=args.eval_to_list,
                                 need_texture=True,
                                 need_fp_fn=args.need_fp_fn)

    # train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.sample_wight, len(train_dataset.sample_wight), 16000)
    # train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.sample_wight, 16000)
    train_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    print("data loader init done")
    return val_loader
