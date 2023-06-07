import numpy as np
from torch.autograd import Function
import torch.nn as nn
import torch
from audtorch.metrics.functional import pearsonr
import torch.nn.functional as F

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


class SimilarityKL(torch.nn.Module):
    def __init__(self, loss_similarity='Cosine', gamma=0.5):
        super(SimilarityKL, self).__init__()
        self.loss_similarity, self.gamma  = loss_similarity, gamma

        if loss_similarity == 'KL':
            self.similarity_loss = F.kl_div
        elif loss_similarity == 'Cosine':
            self.similarity_loss = nn.CosineSimilarity(dim=-1)
        else:
            raise NotImplementedError

    def forward(self, inputs):

        V_F, A_F, T_F = inputs

        # Similarity measured by KL
        if self.loss_similarity == 'KL':
            loss_similarityv_a = 0.5 * (self.similarity_loss(V_F, A_F, reduction='batchmean') + self.similarity_loss(A_F, V_F, reduction='batchmean'))
            loss_similarityv_t = 0.5 * (self.similarity_loss(V_F, T_F, reduction='batchmean') + self.similarity_loss(T_F, V_F, reduction='batchmean'))
            loss_similaritya_t = 0.5 * (self.similarity_loss(T_F, A_F, reduction='batchmean') + self.similarity_loss(A_F, T_F, reduction='batchmean'))
            loss_similarityv_a = 0 if torch.isnan(loss_similarityv_a) else loss_similarityv_a
            loss_similarityv_t = 0 if torch.isnan(loss_similarityv_t) else loss_similarityv_t
            loss_similaritya_t = 0 if torch.isnan(loss_similaritya_t) else loss_similaritya_t

            loss_similarity = (loss_similaritya_t + loss_similarityv_a + loss_similarityv_t) / 3.0

        elif self.loss_similarity == 'Cosine':
            loss_similarityv_a = 1.0 - self.similarity_loss(V_F, A_F).mean()
            loss_similarityv_t = 1.0 - self.similarity_loss(V_F, T_F).mean()
            loss_similaritya_t = 1.0 - self.similarity_loss(A_F, T_F).mean()
            loss_similarity = (loss_similaritya_t + loss_similarityv_a + loss_similarityv_t) / 3.0
        else:
            raise NotImplementedError

        loss_all = loss_similarity
        return loss_all

