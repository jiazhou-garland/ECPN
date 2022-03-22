import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import cv2

def cross_entropy_loss_RCF(inputs, targets, l_weight=1, cuda=2, balance=1.1):
    """
    :param inputs: nx1xhxw
    :param targets: nx1xhxw
    :param balance: const parameter: 1.1
    :param l_weight: weight for balance different layer output
    :return: weighted cross entropy loss  1
    """
    device = torch.device("cuda:"+str(cuda))
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t > 0).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t > 0] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights).to(device)
    # layer combine sigmoid and BCE with weight
    # redunction means average the lose by the batch num
    loss = nn.BCEWithLogitsLoss(weights.float(), reduction='sum')(inputs.float(), targets.float())
    loss = loss * l_weight
    return loss

def distancemap_penalized_loss(inputs, targets, l_weight=1, maskSize=5, cuda=0):
    """
    :param inputs: nx1xhxw
    :param targets: nx1xhxw
    :param l_weight: weight for balance different layer output
    :param maskSize: [int] the size of distance transform kernel
    :param cuda: [int] the available cuda device number
    :return: weighted cross entropy loss  with the mask of distance map
    """

    eps = 1e-9
    device = torch.device("cuda:"+str(cuda))
    n, c, h, w = inputs.size()
    distance_weights = np.zeros((n, h, w))
    tmp = np.zeros((h, w), dtype=np.uint8)

    for i in range(n):

        t = targets[i, :, :, :]
        t = t.squeeze(axis=0)
        t = t.cpu().data.numpy() #nx1xhxw-> hxw

        # inverter of the target
        tmp[np.where(t == 1)] = 0
        tmp[np.where(t == 0)] = 1

        distance_map = cv2.distanceTransform(tmp, cv2.DIST_L2, maskSize=maskSize)
        # print(distance_map)
        distance_map = (distance_map.max() - distance_map) / (distance_map.max()-distance_map.min())
        distance_weights[i, :, :] = distance_map

    # print(distance_weights)
    distance_weights = distance_weights[:, np.newaxis, :, :]
    weights = torch.Tensor(distance_weights).to(device)
    loss = nn.BCEWithLogitsLoss(weights.float(), reduction='mean')(inputs.float(), targets.float())
    loss = loss * l_weight
    return loss

def mixed_loss(inputs, targets, cuda=0):
    loss1 = cross_entropy_loss_RCF(inputs, targets, l_weight=1, cuda=cuda, balance=1.1)
    loss2 = distancemap_penalized_loss(inputs, targets, l_weight=1, maskSize=5, cuda=cuda)
    return loss1 + loss2
