# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/10/7 16:22
# @Ide     : PyCharm
"""

import torch
import torch.nn as nn
import math
import numpy as np
from config import DefaultConfig
from torch.autograd import Variable
import torch.nn.functional as F


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class Bigru(nn.Module):
    def __init__(self, config, vectors=None, target_vectors=None, aspect_vectors=None):
        super(Bigru, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)

        # test
        self.bigru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim,
            bidirectional=True
        )

        # final classifier
        self.fc = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(config.embedding_dim, config.label_num)
        )

    def forward(self, text, target, aspect):
        # text: 50 * 128
        batch_size = text.size()[1]
        text_embed = self.embedding(text)  # 50 * batch * 300
        # kmax的效果不好，改成使用最后一个hidden state
        # out = self.bigru(text_embed)[0].permute(1, 2, 0)  # batch * 600 * 50
        # pooling = kmax_pooling(out, 2, 2)  # batch * 600 * 2
        last_hidden = self.bigru(text_embed)[1]
        last_hidden = last_hidden.permute(1, 0, 2).reshape(batch_size, -1)  # batch * 600

        flatten = last_hidden
        logits = self.fc(flatten)

        return logits

    def get_optimizer(self, lr1, lr2=0):
        embedding_params = map(id, self.embedding.parameters())
        base_params = filter(lambda p: id(p) not in embedding_params, self.parameters())
        optimizer = torch.optim.Adam([
            {'params': self.embedding.parameters(), 'lr': lr2},
            {'params': base_params, 'lr': lr1}
        ])

        return optimizer
