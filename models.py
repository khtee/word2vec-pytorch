"""Pytorch implementation for word embedding models

1. skip-gram
2. GloVe

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class SkipGramModel(nn.Module):
    def __init__(self, num_embed, embed_dim):
        super(SkipGramModel, self).__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.wi = nn.Embedding(num_embed, embed_dim, sparse=True)
        self.wj = nn.Embedding(num_embed, embed_dim, sparse=True)
        # self.bi = nn.Embedding(num_embed, 1)
        # self.bj = nn.Embedding(num_embed, 1)

        initrange = 1.0 / self.embed_dim
        # init.uniform_(self.wi.weight.data, -initrange, initrange)
        # init.constant_(self.wj.weight.data, 0)
        self.wi.weight.data.uniform_(-initrange, initrange)
        self.wj.weight.data.uniform_(-initrange, initrange)
        # self.bi.weight.data.zero_()
        # self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices, neg_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        # b_i = self.bi(i_indices).squeeze()
        # b_j = self.bj(j_indices).squeeze()
        score = torch.sum(torch.mul(w_i, w_j), dim=1)  #+ b_i + b_j
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        emb_neg_v = self.wj(neg_indices)
        neg_score = torch.bmm(emb_neg_v, w_i.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.wi.weight.cpu().data.numpy()
        np.save(file_name, embedding)


class GloveModel(nn.Module):
    def __init__(self, num_embed, embed_dim):
        super(GloveModel, self).__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.wi = nn.Embedding(num_embed, embed_dim, sparse=True)
        self.wj = nn.Embedding(num_embed, embed_dim, sparse=True)
        self.bi = nn.Embedding(num_embed, 1)
        self.bj = nn.Embedding(num_embed, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()

        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j

        return x

    def save_embedding(self, id2word, file_name):
        embedding = self.wi.weight.cpu().data.numpy() + self.wi.weight.cpu(
        ).data.numpy()
        np.save(file_name, embedding)