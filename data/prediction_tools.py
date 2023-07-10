import copy

import cv2
import cv2 as cv
import numpy as np
import os

from .tools import post_processing


# refined_root = '/media/wilson/Wilson/DE/Data/CAPSP_Refined/'
# refined_root = '/home/creator/Data/WilsonData/CAPSP_Refined/'
refined_root = '/data2/Wilson/Data/CAPSP_Refined/'

to_predictions = [
    # 'AD_Net',
    'BDRAR',
    # 'DeshadowNet',
    # 'Direction',
    'Distraction',
    # , 'FastCNN'
    # 'FSD',
    # 'Mask_GAN' ,
    'MTMT',
    # 'FDR',
    # 'SDCM'
    # , 'SD_mobile_robot'
]

init_prediction_order = {
    'BDRAR': 0,
    'Distraction': 1,
    'MTMT': 2,
    # 'FDR': 3,
    # 'SDCM': 3,
    # 'Direction': 4,
}

need_for_modifying = {
    'AD_Net': 'THR',
    'BDRAR': 'NO',
    'DeshadowNet': 'CRF',
    'Direction': 'NO',
    'Distraction': 'NO',
    'FastCNN': 'CRF',
    'FSD': 'NO',
    'MTMT': 'NO',
    'SD_mobile_robot': 'NO'
}


def reading_multi_prediction(rgb_image, rgb_image_name, image_size, need_post_processing=False, is_train=False,
                             synthetic=False, mask=None, resize=False, eval_root=None):
    detections_crf = None
    detections_ori = None
    prediction_refined_multi = None
    # rgb_image.show('rgb_image')

    to_predictions_shuffle = copy.deepcopy(to_predictions)
    if is_train:
        np.random.shuffle(to_predictions_shuffle)
    if not synthetic:
        for idx_pre, method_name in enumerate(to_predictions_shuffle):
            prediction = reading_prediction(method_name, rgb_image_name,eval_root=eval_root)
            # prediction_refined = reading_prediction(method_name, rgb_image_name, prediction_refined=refined_root)

            if resize:
                try:
                    prediction = cv.resize(prediction, (image_size, image_size))
                except:
                    print(method_name, rgb_image_name)
                    exit(-1)

            # prediction_refined = cv.resize(prediction_refined,(image_size, image_size))
            # cv.imshow(method_name, prediction)
            # cv.waitKey(10)

            prediction_ori = np.asarray(prediction)
            # prediction_refined = np.asarray(prediction_refined).astype(np.float) / 255.
            prediction_ori = np.expand_dims(prediction_ori, 0)
            # prediction_refined = np.expand_dims(prediction_refined, 0)

            if need_post_processing:
                prediction_crf = prediction_post_processing(rgb_image, method_name, prediction)
            else:
                prediction_crf = prediction
            prediction_crf = np.asarray(prediction_crf)
            prediction_crf = np.expand_dims(prediction_crf, 0)

            if idx_pre == 0:
                detections_crf = prediction_crf
            else:
                detections_crf = np.concatenate((detections_crf, prediction_crf), 0)
        return detections_crf
    else:
        # create sysentic predictions based on shadow mask
        for i in range(10):
            #
            mask_temp = mask.copy()
            if i < 3:
                kernel = cv.getStructuringElement(cv2.MORPH_RECT, (i * 2 + 1, i * 2 + 1))
                prediction = cv.morphologyEx(mask_temp, cv2.MORPH_OPEN, kernel)
                prediction = np.expand_dims(prediction, 0)
            elif i < 5:
                kernel = cv.getStructuringElement(cv2.MORPH_RECT, (i * 2 + 1, i * 2 + 1))
                prediction = cv.morphologyEx(mask_temp, cv2.MORPH_CLOSE, kernel)
                prediction = np.expand_dims(prediction, 0)
            elif i < 10:
                pointx = np.random.randint(50, image_size, 1, dtype=np.int)
                pointy = np.random.randint(50, image_size, 1, dtype=np.int)
                a = [pointx + (i % 3+ 1) * 5, pointy + (i % 3+ 1) * 5]
                b = [pointx + (i % 3+ 1) * 5, pointy - (i % 3+ 1) * 5]
                c = [pointx - (i % 3 + 1) * 5, pointy - (i % 3+ 1) * 5]
                d = [pointx - (i % 3 + 1) * 5, pointy + (i % 3+ 1) * 5]
                points = np.array([[a, b, c, d]], dtype=np.int32)
                prediction = cv.fillPoly(mask_temp, points, (255))

                pointx = np.random.randint(50, image_size, 1, dtype=np.int)
                pointy = np.random.randint(50, image_size, 1, dtype=np.int)
                a = [pointx + (i % 3+ 1) * 3, pointy + (i % 3+ 1) * 3]
                b = [pointx + (i % 3+ 1) * 3, pointy - (i % 3+ 1 )* 3]
                c = [pointx - (i % 3+ 1) * 3, pointy - (i % 3+ 1 )* 3]
                d = [pointx - (i % 3+ 1) * 3, pointy + (i % 3+ 1) * 3]
                points = np.array([[a, b, c, d]], dtype=np.int32)
                prediction = cv.fillPoly(prediction, points, (255))

                prediction = prediction.transpose(2, 0, 1)
            else:
                prediction = mask

            cv.imshow(str(i), prediction.squeeze(0))
            cv.waitKey(10)

            if i == 0:
                detections = prediction
            else:
                detections = np.concatenate((detections, prediction), 0)

            np.random.shuffle(detections)
        cv.imshow('mask', mask)
        cv.waitKey(0)
        return detections


def reading_multi_refined_prediction(rgb_image_name, image_size, is_train=False, eval_root=None):
    detections_ori = None
    # rgb_image.show('rgb_image')

    to_predictions_shuffle = to_predictions
    if is_train:
        np.random.shuffle(to_predictions_shuffle)
    for idx_pre, method_name in enumerate(to_predictions_shuffle):
        prediction = reading_prediction(method_name, rgb_image_name, prediction_refined=refined_root, eval_root=eval_root)
        prediction = cv.resize(prediction, (image_size, image_size))

        # cv.imshow(method_name, prediction)
        # cv.waitKey(10)
        prediction_ori = np.asarray(prediction).astype(np.float) / 255.
        prediction_ori = np.expand_dims(prediction_ori, 0)

        if idx_pre == 0:
            detections_ori = prediction_ori
        else:
            detections_ori = np.concatenate((detections_ori, prediction_ori), 0)

    return detections_ori


def reading_prediction(method_name_temp, img_name, img_from_list_temp=True, prediction_refined=None, eval_root=None):
    if prediction_refined == None:
        prediction_root = os.path.join(eval_root, method_name_temp)
    else:
        prediction_root = prediction_refined
    # reading image name from list or dir
    if img_from_list_temp:
        if prediction_refined == None:
            prediction_name = os.path.join(prediction_root, img_name)
        else:
            prediction_name = os.path.join(prediction_root,
                                           '%s_%d.png' % (img_name[:-4], init_prediction_order[method_name_temp]))
    else:
        prediction_name = os.path.join(prediction_root, os.path.splitext(img_name)[0] + '.png')

    if os.path.exists(prediction_name) is False:
        prediction_name = prediction_name[:-4] + '.png'
        if os.path.exists(prediction_name) is False:
            if not prediction_refined == None:
                print('Refined')
            print(method_name_temp, ' prediction file ', img_name, 'do not exit')
            # error_info.update({str(name + img_name): str(method_name_temp)})
            return reading_prediction('BDRAR', img_name)

    # img_set.update({str(method_name): str(prediction_name)})
    prediction = cv.imread(prediction_name, cv.IMREAD_GRAYSCALE)
    return prediction


def prediction_post_processing(ori_img_temp, method_name_temp, prediction):
    # Post processing
    signal = need_for_modifying[method_name_temp]
    if signal == 'CRF':
        prediction = post_processing(ori_img_temp, prediction, processing_method='CRF')
    elif signal == 'THR':
        prediction = post_processing(ori_img_temp, prediction, processing_method='THR')
    return prediction
