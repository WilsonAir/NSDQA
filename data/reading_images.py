import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
import cv2 as cv


def find_image_file(image_name):
    if not os.path.exists(image_name + '.png'):
        if not os.path.exists(image_name + '.jpg'):
            if not os.path.exists(image_name + '.JPG'):
                print(image_name + '.jpg' + ' not exist！')
                return 'Can not find file!'
            else:
                image_name = image_name + '.JPG'
        else:
            image_name = image_name + '.jpg'
    else:
        image_name = image_name + '.png'
    return image_name


def reading_images(root, path, image_size, file_format=None, channel=3, normallize=False, resize=False):
    # for shadow mask information
    image_name = os.path.join(root, path)
    if file_format:
        image_name += file_format
    if channel == 1:
        image_ori = cv.imread(image_name, cv.IMREAD_GRAYSCALE)
        ori_size = image_ori.shape
        if resize:
            image_ori = cv.resize(image_ori, (image_size, image_size))
        image = np.expand_dims(image_ori, 2)
    else:
        image_ori = cv.imread(image_name, cv.IMREAD_COLOR)
        ori_size = image_ori.shape
        image = cv.resize(image_ori, (image_size, image_size))

    if normallize:
        image = image.astype(np.float)/255.

    return image, ori_size


def read_texture_info(img_name, new_img_name, image_size, base_name, sign):
    # for Texture information
    if str(img_name).find('CUHKshadow') >= 0:
        img_name =  str(img_name).replace("/shadow_", "/mask_")
    if str(new_img_name).find('CUHKshadow') >= 0:
        new_img_name =  str(new_img_name).replace("/shadow_", "/mask_")

    segpath = img_name + '_seg.mat'
    if sign == 1:
        singlepath = new_img_name + '_single.mat'
        sign = 0
    else:
        singlepath = new_img_name[:-2] + '_single.mat'
    seg = loadmat(segpath)['seg']
    feature = loadmat(singlepath)['testdata']
    newseg = ndarray_nearest_neighbour_scaling(seg, image_size, image_size)
    try:
        texture_feature = feature[newseg[:, :] - 1]
    except:
        print(base_name, segpath, singlepath)
    return texture_feature


def ndarray_nearest_neighbour_scaling(label, new_h, new_w):
    """
    Implement nearest neighbour scaling for ndarray
    :param label: [H, W] or [H, W, C]
    :return: label_new: [new_h, new_w] or [new_h, new_w, C]
    Examples
    --------
    ori_arr = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=np.int32)
    new_arr = ndarray_nearest_neighbour_scaling(ori_arr, new_h=8, new_w=10)
    >> print(new_arr)
    [[1 1 1 1 2 2 2 3 3 3]
     [1 1 1 1 2 2 2 3 3 3]
     [1 1 1 1 2 2 2 3 3 3]
     [4 4 4 4 5 5 5 6 6 6]
     [4 4 4 4 5 5 5 6 6 6]
     [4 4 4 4 5 5 5 6 6 6]
     [7 7 7 7 8 8 8 9 9 9]
     [7 7 7 7 8 8 8 9 9 9]]
    """
    if len(label.shape) == 2:
        label_new = np.zeros([new_h, new_w], dtype=label.dtype)
    else:
        label_new = np.zeros([new_h, new_w, label.shape[2]], dtype=label.dtype)

    scale_h = new_h / label.shape[0]
    scale_w = new_w / label.shape[1]

    y_pos = np.arange(new_h)
    x_pos = np.arange(new_w)
    y_pos = np.floor(y_pos / scale_h).astype(np.int)
    x_pos = np.floor(x_pos / scale_w).astype(np.int)

    y_pos = y_pos.reshape(y_pos.shape[0], 1)
    y_pos = np.tile(y_pos, (1, new_w))
    x_pos = np.tile(x_pos, (new_h, 1))
    assert y_pos.shape == x_pos.shape

    label_new[:, :] = label[y_pos[:, :], x_pos[:, :]]
    return label_new