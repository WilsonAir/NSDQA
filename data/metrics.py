""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from .prediction_tools import to_predictions
from scipy import stats
from ignite.metrics.metric import Metric


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True) #Return: values, indices = a.topk()
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def compute_IOU(im_pred, im_lab):
    im_pred = np.asarray(im_pred, dtype=np.float64).copy()
    im_lab = np.asarray(im_lab, dtype=np.float64).copy()

    overlap_t = np.sum((im_pred > 0) * (im_lab > 0))
    union_t = np.sum((im_pred + im_lab) > 0)

    # im_pred = np.ones(im_pred.shape)-im_pred
    # im_lab = np.ones(im_lab.shape)-im_lab

    # overlap_f = np.sum((im_pred > 0) * (im_lab > 0))
    # union_f = np.sum((im_pred + im_lab) > 0)

    try:
        if union_t == 0:
            iou = 0
        else:
            iou = (overlap_t / union_t)  # + overlap_f / union_f)/2
    except:
        iou = 0.5
        # print(overlap_t, union_t )#, overlap_f , union_f)
    return iou


def calcuber_single(pre, gt, image_name=None):
    if np.max(pre) > 10:
        thre = 127
    else:
        thre = 0.5

    N_p = np.sum(gt > thre)
    N_n = np.sum(gt <= thre)
    T_p = np.sum((gt > thre) * (pre > thre))
    T_n = np.sum((gt <= thre) * (pre <= thre))

    if N_p == 0:
        acc_shadow = 0.0
    else:
        acc_shadow = T_p / N_p
    if N_n == 0:
        acc_nonsha = 0.0
    else:
        acc_nonsha = T_n / N_n

    shadow = 100 * (1 - acc_shadow)
    nonshadow = 100 * (1 - acc_nonsha)
    ber = 100 * (1 - (acc_shadow + acc_nonsha) / 2)

    # shadow = acc_shadow
    # nonshadow = acc_nonsha
    # ber = (acc_shadow + acc_nonsha) / 2

    beta = 0.3

    # precision = T_p / np.sum(pre > thre)
    # recall = T_p / (T_p+np.sum((gt > thre) * (pre < thre)))
    # S_f = (1+beta**2)*precision*recall/(beta**2 * precision + recall)

    # thre = np.sum(pre) / (pre.shape[0] * pre.shape[1])
    # precision = T_p / np.sum(pre > thre)
    # recall = T_p / (T_p + np.sum((gt > thre) * (pre < thre)))
    # ADAPF = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    ADAPF = 0
    S_f = 0
    # if precision == 0 or recall == 0:
    #     file = open('output/test.txt', 'a')
        # print(image_name)
        # file.writelines(image_name+'\n')
        # file.close()
        # print(np.sum(pre>thre))
        # print(np.sum(gt>thre))
        # cv.imshow('pre',pre)
        # cv.imshow('gt',gt)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    return shadow, nonshadow, ber, S_f, ADAPF


def calcuber(pre, gt, Np, Nn, Tp, Tn):
    if np.max(pre) > 10:
        thre = 127
    else:
        thre = 0.5

    Np += np.sum(gt > thre)
    Nn += np.sum(gt < thre)
    Tp += np.sum((gt > thre) * (pre > thre))
    Tn += np.sum((gt < thre) * (pre <= thre))
    if Np == 0:
        acc_shadow = 0.0
    else:
        acc_shadow = Tp / Np
    if Nn == 0:
        acc_nonsha = 0.0
    else:
        acc_nonsha = Tn / Nn

    shadow = 100 * (1 - acc_shadow)
    nonshadow = 100 * (1 - acc_nonsha)
    ber = 100 * (1 - (acc_shadow + acc_nonsha) / 2)

    return shadow, nonshadow, ber, Np, Nn, Tp, Tn

def cal_acc(prediction, label, thr = 128):
    prediction = (prediction > thr)
    label = (label > thr)
    prediction_tmp = prediction.astype(np.float)
    label_tmp = label.astype(np.float)
    TP = np.sum(prediction_tmp * label_tmp)
    TN = np.sum((1 - prediction_tmp) * (1 - label_tmp))
    Np = np.sum(label_tmp)
    Nn = np.sum((1-label_tmp))
    Union = np.sum(prediction_tmp) + Np - TP

    return TP, TN, Np, Nn, Union


def calcuber_with_dict(pre, gt, selection_result):
    if np.max(pre) > 10:
        thre = 128
    else:
        thre = 0.5

    Np = selection_result['Np']
    Nn = selection_result['Nn']
    Tp = selection_result['Tp']
    Tn = selection_result['Tn']

    # TP_single, TN_single, Np_single, Nn_single, Union = cal_acc(pre, gt)
    # Tp = Tp + TP_single
    # Tn = Tn + TN_single
    # Np = Np + Np_single
    # Nn = Nn + Nn_single

    Np += np.sum(gt > thre)
    Nn += np.sum(gt <= thre)
    Tp += np.sum((gt > thre) * (pre > thre))
    Tn += np.sum((gt <= thre) * (pre <= thre))

    if Np == 0:
        acc_shadow = 0.0
    else:
        acc_shadow = Tp / Np
    if Nn == 0:
        acc_nonsha = 0.0
    else:
        acc_nonsha = Tn / Nn

    shadow = 100 * (1 - acc_shadow)
    nonshadow = 100 * (1 - acc_nonsha)
    ber = 100 * (1 - (acc_shadow + acc_nonsha) / 2)

    return {'shadow': shadow, 'nonshadow': nonshadow, 'ber': ber, 'Np': Np, 'Nn': Nn, 'Tp': Tp, 'Tn': Tn}


def measure_bers(input, output, detection_results, mask, mask_size, selection_result):
    # measure ber
    for idx in range(output.shape[0]):
        method_number_ith = F.softmax(output[idx]).argmax(dim=0).cpu()
        pre_ori = np.asarray(detection_results[idx][method_number_ith])

        # cv.imshow('prediction',pre)
        # cv.imshow('ori_img',ori_img)
        # cv.imshow('prediction_ori',pre_ori)
        # cv.waitKey(0)

        gt = np.asarray(mask[idx]).squeeze(2)
        selection_result_ = calcuber_with_dict(pre_ori, gt, selection_result)
    return selection_result_


def measure_bers_for_ori(detection_results, mask, mask_size, detection_eval):
    # measure ber
    for idx in range(detection_results.shape[0]):
        ber_minest = 100
        best_pre = None
        gt = np.asarray(mask[idx]).squeeze(2)

        for idx2 in range(len(to_predictions)):
            method_name = to_predictions[idx2]
            pre = np.asarray(detection_results[idx][idx2])

            single_eval = calcuber_with_dict(pre, gt, detection_eval[method_name])
            detection_eval.update({method_name: single_eval})

            shadow, nonshadow, ber, S_f = calcuber_single(pre, gt)
            if ber < ber_minest:
                best_pre = pre
                ber_minest = ber

        best_single_eval = calcuber_with_dict(best_pre, gt, detection_eval['best'])
        detection_eval.update({'best': best_single_eval})

    return detection_eval


class IQAPerformance():
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.
    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self):
        self._y_pred = []
        self._y = []
        self._y_std = []

    def update(self, pred, y):
        self._y.append(np.asarray(y.cpu()))
        # self._y_std.append(y[1].item())
        self._y_pred.append(np.asarray(pred.detach().cpu()))

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        # sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))

        # print(sq.shape)
        # print(q.shape)

        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()
        # outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

        return srocc, krocc, plcc, rmse, mae, #outlier_ratio