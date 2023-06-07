import numpy as np
from torch.autograd import Function
import torch.nn as nn
import torch
from audtorch.metrics.functional import pearsonr
import torch.nn.functional as F
"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""


class cos_loss(nn.Module):
    def __init__(self):
        super(cos_loss, self).__init__()

    def forward(self, p_v, y, y_pred):
        pos_index = np.array([i for i, e in enumerate(y) if e >= 0])
        neg_index = np.array([i for i, e in enumerate(y) if e < 0])

        if len(pos_index) != 0:
            pos = p_v[torch.from_numpy(pos_index)]
            pos_avgvec = torch.mean(pos, dim=0).unsqueeze(0)
        if len(neg_index) != 0:
            neg = p_v[torch.from_numpy(neg_index)]
            neg_avgvec = torch.mean(neg, dim=0).unsqueeze(0)

        pos_index_pred = np.array([i for i, e in enumerate(y_pred) if e >= 0])
        neg_index_pred = np.array([i for i, e in enumerate(y_pred) if e < 0])

        if len(pos_index_pred) != 0:
            pos_pred = p_v[torch.from_numpy(pos_index_pred)]
            pos_avgvec_pred = torch.mean(pos_pred, dim=0).unsqueeze(0)
        if len(neg_index_pred) != 0:
            neg_pred = p_v[torch.from_numpy(neg_index_pred)]
            neg_avgvec_pred = torch.mean(neg_pred, dim=0).unsqueeze(0)

        if len(pos_index) != 0 and len(pos_index_pred) != 0:
            cos_pos = 1 - torch.cosine_similarity(pos_avgvec, pos_avgvec_pred)
            # cos_pos =  F.pairwise_distance(pos_avgvec, pos_avgvec_pred, p=2)
        else:
            cos_pos = 0

        if len(neg_index) != 0 and len(neg_index_pred) != 0:
            cos_neg = 1 - torch.cosine_similarity(neg_avgvec, neg_avgvec_pred)
            # cos_neg = F.pairwise_distance(neg_avgvec, neg_avgvec_pred, p=2)
        else:
            cos_neg = 0

        polar_loss = len(pos_index) * cos_pos / len(y) +  len(neg_index) * cos_neg / len(y)
        return polar_loss

class corr_loss(nn.Module):
    def forward(self,tensor_1,tensor_2):
        def minmaxscaler(data):
            min = torch.min(data)
            max = torch.max(data)
            return (data - min) / (max - min)
        x =  minmaxscaler(abs(tensor_1))
        y =  minmaxscaler(abs(tensor_2)).squeeze(1)

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        p1 = torch.sum(vx.mul(vy))
        p2 = ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))) + 1e-4

        cost =  p1 / p2
        return 1 - cost
        # return F.pairwise_distance(x,y)

class CosineSimilarity(nn.Module): # Jack Add 0521
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, tensor1, tensor2):
        dot_product = torch.sum(tensor1 * tensor2, dim=1)
        norm1 = torch.norm(tensor1, dim=1)
        norm2 = torch.norm(tensor2, dim=1)
        similarity = dot_product / (norm1 * norm2)
        similarity_mean = torch.mean(similarity) # 取余弦相似性的均值，作为单独的数值输出
        return similarity_mean



class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss




