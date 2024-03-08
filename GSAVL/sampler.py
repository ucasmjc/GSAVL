import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from numpy.random import choice
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def samplers(type):
    samplers = {
        'naive': naive_sampler,
        'grid': grid_sampler,
        'prob':prob_sampler,
    }
    return samplers[type]

def naive_sampler(map):
    #找到特征图中最大点的位置，暴力上采样得到坐标
    cor = np.unravel_index(np.argmax(map[:, :]), map[:, :].shape)
    point = np.zeros((1, 2))
    point[0, 0] = cor[1]
    point[0, 1] = cor[0]
    return point

def grid_sampler(map):
    max_y, max_x = np.unravel_index(np.argmax(map[:, :]), map[:, :].shape)
    max_point = np.array([[max_x, max_y]])
    dist=10
    # 找到相邻点
    adjoin_x = [max(0, max_x - dist), max_x, min(max_x + dist, map.shape[1] - 1)]
    adjoin_y = [max(0, max_y - dist), max_y, min(max_y + dist, map.shape[0] - 1)]
    nearest_point=None
    nearest=0
    for x in adjoin_x:
        for y in adjoin_y:
            if x == max_x and y == max_y:
                continue
            if map[y,x]>nearest:
                nearest_point=np.array(([[x,y]]))
                nearest=map[y,x]

    points = np.concatenate((max_point, nearest_point),axis=0)
    return points


def prob_sampler(map):
    N=3
    prob=0.01
    sorted_indices = np.argsort(map.flatten())
    top_indices = sorted_indices[int((1-prob) * len(sorted_indices)):]
    cors_y,cors_x = np.unravel_index(top_indices, map.shape)
    probabilities = map[cors_y,cors_x]
    sampled_indices = np.random.choice(cors_x.shape[0], size=N, replace=True, p=probabilities / np.sum(probabilities))
    points =np.zeros((N,2))
    for i,indice in enumerate(sampled_indices):
        points[i][0]=cors_x[indice]
        points[i][1] = cors_y[indice]
    return points

