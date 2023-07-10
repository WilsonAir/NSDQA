import copy
import os
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage

from .metrics import calcuber_single, compute_IOU
from .prediction_tools import reading_multi_prediction
from .reading_images import reading_images, read_texture_info
# from .boundary_iou import boundary_iou

# from timm.utils.evaluator import Eval_single_thread
# from .models.layers.MR8 import apply_filterbank, makeRFSfilters
from PIL import Image


class TextureDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, phase='train', transform=None, augment=False, image_size=64, multi_path=1,
                 dataset='All', dataset_path='', detections_path='', eval_to_list=False, need_texture=False, synthetic=False, need_fp_fn=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            root_dir(string): Path to texture files

        """
        self.ori_lines = []                 # shadow image and label before selected
        self.landmarks_frame = []           # shadow image and label afer selected
        self.shadow_mask_list = []          # shadow mask
        self.shadow_free = []               # shadow free image

        self.sample_wight = []
        self.label = []

        # For the detection result
        self.add_detection_res = True
        self.detection_result_list = None

        # To choose which dataset for training
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.detections_path = detections_path
        # self.root_dir = root_dir
        self.transform = transform
        self.multi_path = multi_path
        self.image_size = image_size
        self.eval_to_list = eval_to_list
        self.data_augment = augment

        # For prediction fp fn
        self.need_fp_fn = need_fp_fn

        if phase == 'train':
            self.train = True
            bad_label = open('output/test.txt').readlines()
        else:
            self.train = False
            if phase == 'train_val':
                self.train_val = True
        self.synthetic = synthetic



        if csv_file[-4:] == '.txt':
            temp = []
            lines = open(csv_file).readlines()
            for idx, line in enumerate(lines):
                temp_line = line.strip('\n').split('\t')
                if self.train:
                    if temp_line[0]+'\n' in bad_label:
                        a = 1
                        temp.append(temp_line)
                    else:
                        temp.append(temp_line)
                else:
                    temp.append(temp_line)
            self.ori_lines = temp
        elif csv_file[-4:] == '.csv':
            self.ori_lines = pd.read_csv(csv_file, encoding='utf8', header=None, sep=',').values.tolist()

        self.split_dataset()

        print("{} dataset init donw！ total {} images".format('Train' if self.train else 'Test', len(self.landmarks_frame)))

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        base_name_ori = self.landmarks_frame[idx][0]
        base_name = base_name_ori
        if base_name[-4:] == '.jpg' or base_name[-4:] == '.png' or base_name[-4:] == '.JPG':
            base_name = base_name[:-4]
        sign = 0
        if not base_name[:17] == 'Data_augmentation':
            base_name = 'Data_augmentation/' + base_name
            sign = 1

        # for RGB information
        rgb_image, rgb_size = reading_images(self.dataset_path, base_name_ori, self.image_size, channel=3)

        resize = False
        if self.train:
            resize = True

        mask, mask_size = reading_images(os.path.dirname(self.shadow_mask_list[idx]), os.path.basename(self.shadow_mask_list[idx]), self.image_size, file_format='.png', channel=1, resize=resize)

        fp, fn = None, None
        if self.need_fp_fn:
            fp, _ = reading_images('data/dataset/errors/', base_name_ori[:-4], self.image_size, file_format='_fp.png', channel=1, resize=resize)
            fn, _ = reading_images('data/dataset/errors/', base_name_ori[:-4], self.image_size, file_format='_fp.png', channel=1, resize=resize)
            # false, mask_size = reading_images('output/', base_name_ori[:-4], self.image_size, file_format='.png', channel=1, resize=resize)

        newimg = rgb_image


        # for detection result
        detection_results_crf = reading_multi_prediction(rgb_image, base_name_ori, self.image_size, is_train=self.train, resize=resize, eval_root=self.detections_path)
        target = get_label(base_name_ori, detection_results_crf, mask)


        targets = {
            'image_name':  base_name_ori,
            'rgb_image': rgb_image,
            'label_ber': target['label_ber'],
            'label_biou': target['label_biou'],
            'label_shadow': target['label_shadow'],
            'bers': target['bers'],
            'boundary_ious': target['boundary_ious'],
            'ious': target['ious'],
            'shadows': target['shadows'],
            'nonshadows': target['nonshadows'],
            'fmeasure': target['fmeasure'],
            'maes': target['maes'],
            'ADAPFs': target['ADAPFs']
        }

        if self.transform:
            ## Image Net
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            normallize = torchvision.transforms.Normalize(mean, std)

            from data.joint_transform import RandomHorizontallyFlip, RandomVertFlip, Enhance, To_pil, To_tensor, Resize, Compose
            if self.train:
                joint_transform = Compose([
                    To_pil(),
                    RandomHorizontallyFlip(),
                    RandomVertFlip(),
                    # Enhance(),
                    To_tensor()]
                )
            else:
                joint_transform = Compose([
                    To_pil(),
                    Resize((self.image_size, self.image_size)),
                    To_tensor()]
                )

            if not self.need_fp_fn:
                newimg, mask_resized, detection_results_for_train = joint_transform(newimg, mask, detection_results_crf)
            else:
                newimg, mask_resized, detection_results_for_train, fp, fn = joint_transform(newimg, mask,
                                                                                        detection_results_crf, fp, fn)
            detection_results_for_train = torch.cat(detection_results_for_train, dim=0)
            newimg_before_norm = copy.deepcopy(newimg)
            newimg = normallize(newimg)


        targets.update({'detection_results_crf_for_train': detection_results_for_train})
        targets.update({'detection_results_crf': detection_results_crf})
        targets.update({'mask_resized': mask_resized})
        targets.update({'mask': mask})
        targets.update({'img': newimg_before_norm})
        targets.update({'ori_size': torch.tensor((mask_size[0],mask_size[1]), dtype=int)})

        if self.need_fp_fn:
            targets.update({'fp': fp})
            targets.update({'fn': fn})
        return newimg, targets

    # Split datasets to SBU, ISTD, CUHK, all
    def split_dataset(self):
        # For the shadow mask
        shadow_mask_list_all = []

        # to split datasets
        for idx in range(len(self.ori_lines)):
            item = self.ori_lines[idx]
            rgb_image_path = item[0]
            rgb_image_name = os.path.join(self.dataset_path, rgb_image_path)

            if rgb_image_name.find(self.dataset):
                if not os.path.exists(rgb_image_name):
                    print(rgb_image_name, ' not exist！')

            shadow_mask_name = str(rgb_image_name).replace('img', 'label')
            shadow_mask_list_all.append(shadow_mask_name[:-4])

        self.landmarks_frame = self.ori_lines
        self.shadow_mask_list = shadow_mask_list_all



def get_label(image_name, detection_results, mask):
    # measure ber
    bers = []
    shadows = []
    nonsha = []
    ious = []
    fmeasure = []
    ADAPFs = []
    maes = []
    boundary_ious = []
    # smeasure = []
    # fuse_metric = []

    # evaluator = Eval_single_thread(cuda=False)
    for idx in range(detection_results.shape[0]):
        pre = detection_results[idx]
        gt = mask.squeeze()
        shadow, nonshadow, ber, S_f, ADAPF = calcuber_single(pre, gt, image_name=image_name)
        iou = compute_IOU(pre, gt)
        mae = np.sum(np.abs(pre-gt)) / (pre.shape[0] * pre.shape[1])
        # biou = boundary_iou(gt, pre)
        biou = 0

        # res = evaluator.run(pre, gt)
        # sm = res['Sm']
        bers.append(ber)
        ious.append(iou)
        boundary_ious.append(biou)

        fmeasure.append(ADAPF)
        maes.append(mae)
        ADAPFs.append(S_f)
        shadows.append(shadow)
        nonsha.append(nonshadow)
        # smeasure.append(sm)
        # fuse = 100*(1.0 - sm) + ber
        # fuse_metric.append(fuse)
    # cv.imshow('mask', mask)
    # cv.waitKey(10)
    label_ber = np.asarray(bers).argmin()
    # label_ber = np.asarray(bers).argmax()
    label_biou = np.asarray(boundary_ious).argmax()
    label_fmaesure = np.asarray(fmeasure).argmax()
    label_shadow = np.asarray(shadows).argmin()
    # label_nonsha = np.asarray(nonsha).argmin()
    # label_iou = np.asarray(ious).argmin()
    # label_sm = np.asarray(smeasure).argmax()
    # label_fuse = np.asarray(fuse_metric).argmin()
    # if not label_sm == label_ber:
    #     label_sm =label_fuse
    # return label_shadow, label_nonsha, label_ber, label_iou, label_sm

    target = {
        'label_ber': label_ber.astype(np.float),
        'label_biou': label_biou.astype(np.float),
        'label_fmaesure': label_fmaesure.astype(np.float),
        'label_shadow': label_shadow.astype(np.float),
        'bers': np.asarray(bers).astype(np.float),
        'boundary_ious': np.asarray(boundary_ious).astype(np.float),
        'ious': np.asarray(ious).astype(np.float),
        'shadows': np.asarray(shadows).astype(np.float),
        'nonshadows': np.asarray(nonsha).astype(np.float),
        'fmeasure': np.asarray(fmeasure).astype(np.float),
        'maes': np.asarray(maes).astype(np.float),
        'ADAPFs': np.asarray(ADAPFs).astype(np.float)
        }

    return target


class toTensor(object):
    def __call__(self, img):
        img = np.asarray(img)
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img).type(torch.float)


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        img = sample['image']

        if not self.inplace:
            img = img.clone()

        dtype = img.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=img.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=img.device)
        img.sub_(mean[:, None, None]).div_(std[:, None, None] + (std[:, None, None] == 0) * 1.)
        sample['image'] = img
        return sample


if __name__ == '__main__':
    seg = loadmat(
        '/media/wilson/Wilson/DE/Data/Texture_dataset/ori_color_texture/Data_augmentation/SRD/train_data/shadow/_MG_6152_1_seg.mat')[
        'seg']
    single = loadmat(
        '/media/wilson/Wilson/DE/Data/Texture_dataset/ori_color_texture/Data_augmentation/SRD/train_data/shadow/_MG_6152_1_single.mat')[
        'testdata']

    train_path = '/home/wilson/DE/Python/pytorch_shake_shake-master/ShadowDataset/texture_3_class_train.csv'
    test_path = '/home/wilson/DE/Python/pytorch_shake_shake-master/ShadowDataset/texture_3_class_test.csv'
    train_root = '/media/wilson/Wilson/DE/Data/Texture_dataset/ori_color_texture/'
    test_root = '/media/wilson/Wilson/DE/Data/Texture_dataset/ori_color_texture/'
    train_dataset = TextureDataset(train_path, train_root,
                                   transform=transforms.Compose([toTensor()]), multi_path=2, only_use_sbu=True)
    #                                                             , Normalize(mean, std)
    test_dataset = TextureDataset(test_path, test_root,
                                  transform=transforms.Compose([toTensor()]), multi_path=2, only_use_sbu=True)
    #                                                             , Normalize(mean, std)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,

    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=False,

    )

    for step, (data, targets) in enumerate(train_loader):

        train_image = data
        x, x_hsv = data.split(128, 1)
        for i in range(128):
            x_image = ToPILImage()(x[1][i])
            plt.figure(figsize=(64, 64))
            plt.imshow(x_image)

            # mngr = plt.get_current_fig_manager()
            # mngr.window.wm_geometry("+380+310")  # 调整窗口在屏幕上弹出的位置
            plt.pause(1)  # 该句显示图片5秒
            plt.ioff()  # 显示完后一定要配合使用plt.ioff()关闭交互模式，否则可能出奇怪的问题

            plt.clf()  # 清空图片
            plt.close()

        x_hsv, x_hsv_img = x_hsv.split(63, 1)

    print(np.max(seg))
    print(np.min(seg))
    print(len(single))
