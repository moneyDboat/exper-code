# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/10/3 16:09
# @Ide     : PyCharm
"""
import torch
import torch.nn as nn
import math
import numpy as np
from config import DefaultConfig
import torch.nn.functional as F


class EntNet(nn.Module):
    def __init__(self, config, vectors=None, target_vectors=None, aspect_vectors=None):
        super(EntNet, self).__init__()
        self.config = config

        # text, target和aspect各自的embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        if vectors is not None:
            self.embedding.weight.data.copy_(vectors)
        self.target_embedding = nn.Embedding(config.target_vocab_size, config.embedding_dim)
        if target_vectors is not None:
            self.target_embedding.weight.data.copy_(target_vectors)
        self.aspect_embedding = nn.Embedding(config.aspect_vocab_size, config.embedding_dim)
        if aspect_vectors is not None:
            self.aspect_embedding.weight.data.copy_(aspect_vectors)

        tired_keys = torch.LongTensor([1, 2])  # ['Location1', 'Location2']
        tired_keys_embed = self.target_embedding(tired_keys)  # 2 * 300
        # 5*300，训练过程中会不断更新
        self.free_keys_embed = nn.Parameter(
            torch.randn((self.config.n_keys - self.config.n_tied), self.config.embedding_dim))
        keys_emb = torch.cat([tired_keys_embed, self.free_keys_embed], dim=0)  # 7* 300
        self.keys = keys_emb

        # 前向和后向
        self.ent_cell_fw = EntNetCell(config.n_keys, config.embedding_dim, self.keys)
        self.ent_cell_bw = EntNetCell(config.n_keys, config.embedding_dim, self.keys)


        # final classifier
        self.W_att = nn.Parameter(torch.randn(config.embedding_dim, config.embedding_dim * 2))  # 300*600
        self.c1 = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.bn = nn.BatchNorm1d(config.embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.c2 = nn.Linear(config.embedding_dim, config.label_num)

    def forward(self, text, target, aspect):
        # text: 50 * 128
        batch_size = text.size()[1]
        text_embed = self.embedding(text)  # 50 * 128 * 300
        target_embed = self.embedding(target)  # 128 * 300
        aspect_embed = self.embedding(aspect)  # 128 * 300

        # 参数初始化
        h_fw = torch.randn(self.config.n_keys, batch_size, self.config.embedding_dim)  # 7 * 128 * 300
        b_fw = torch.randn(self.config.n_keys, batch_size, self.config.embedding_dim)  # 7 * 128 * 300
        if self.config.cuda:
            h_fw, b_fw = h_fw.cuda(), b_fw.cuda()
        h_bw = torch.randn(batch_size, self.config.embedding_dim)
        b_bw = torch.randn(batch_size, self.config.embedding_dim)
        for text in text_embed:
            h_fw, b_fw = self.ent_cell_fw(text, h_fw, b_fw)
            # 双向后面再改
            # h_bw, b_bw = self.ent_cell_bw(h_bw, b_bw)

        # p_j = softmax(k_j W_att [t a])
        last_h_fw = torch.cat([h.unsqueeze(1) for h in h_fw], dim=1)  # 128 * 7 * 300
        target_aspect = torch.cat([target_embed, aspect_embed], dim=1)  # 128 * 600

        att = torch.mm(self.keys, self.W_att)
        att = torch.mm(att, target_aspect.permute(1, 0)).permute(1, 0)  # 128 * 7
        att = F.softmax(att, dim=1)

        u = torch.sum(last_h_fw * att.unsqueeze(2), dim=1)  # 128 * 300

        # y = softmax(R (H u + a))
        hidden = self.c1(u)
        hidden = hidden + aspect_embed
        hidden = self.relu(self.bn(hidden))
        logit = self.c2(hidden)  # 128 * 2

        return logit

    def get_optimizer(self, lr1, lr2=0):
        embedding_params = list(map(id, self.embedding.parameters())) + list(
            map(id, self.target_embedding.parameters())) + list(map(id, self.aspect_embedding.parameters()))
        base_params = filter(lambda p: id(p) not in embedding_params, self.parameters())
        optimizer = torch.optim.Adam([
            {'params': self.embedding.parameters(), 'lr': lr2},
            {'params': self.target_embedding.parameters(), 'lr': lr2},
            {'params': self.aspect_embedding.parameters(), 'lr': lr2},
            {'params': base_params, 'lr': lr1}
        ])

        return optimizer


class EntNetCell(nn.Module):
    def __init__(self, num_blocks, num_unit_per_block, keys):
        super(EntNetCell, self).__init__()
        self.num_blocks = num_blocks
        self.num_unit_per_block = num_unit_per_block  # 300
        self.keys = keys

        self.U = nn.Parameter(torch.randn(self.num_unit_per_block, self.num_unit_per_block))
        self.V = nn.Parameter(torch.randn(self.num_unit_per_block, self.num_unit_per_block))
        self.W = nn.Parameter(torch.randn(self.num_unit_per_block, self.num_unit_per_block))
        self.U_bias = nn.Parameter(torch.randn(self.num_unit_per_block))
        self.v = nn.Parameter(torch.randn(self.num_unit_per_block))

        self.gru_cell = nn.GRUCell(self.num_unit_per_block, self.num_unit_per_block, bias=True)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, w, h, d):
        # w: batch * 300
        # h/d: 7 * batch * 300
        batch = w.size()[0]
        new_h = []
        new_d = []
        # 对2+5个记忆单元各进行一次计算
        for j, h_j in enumerate(h):
            k_j = self.keys[j]
            candi_j = self.get_candidate(h_j, k_j, w, batch)  # batch * 300

            new_d_j = self.gru_cell(candi_j, d[j])
            new_d.append(new_d_j)

            gate_j = self.get_gate(h_j, k_j, w, new_d_j)

            # Equation 5: h_j <- h_j + g_j * h_j^~
            # perform an update of the hidden state (memory)
            h_j_next = h_j + gate_j.unsqueeze(dim=1) * candi_j  # batch * 300

            # Equation 6: h_j <- h_j / \norm{h_j}
            # Forget previous memories by normalization
            h_j_next_norm = torch.norm(h_j_next, dim=1)
            h_j_next = h_j_next / h_j_next_norm.unsqueeze(dim=1)
            new_h.append(h_j_next)

        return new_h, new_d

    def get_candidate(self, h_j, k_j, w, batch):
        '''
        Represents the new memory candidata that will be weighted by the
        gate value and combined with the existing memory.
        Equation 3:
        h_j^~ <- \phi(U h_j + V k_j + W w)
        '''
        k_j = k_j.repeat(batch, 1)  # batch * 300
        h_U = torch.mm(h_j, self.U) + self.U_bias  # batch * 300
        k_V = torch.mm(k_j, self.V)
        w_W = torch.mm(w, self.W)
        return F.relu(h_U + k_V + w_W)

    def get_gate(self, h_j, k_j, w, d_j):
        '''
        Implements the gate (scalar for each block).
        Equation 2:
        g_j <- \sigma(w_i^T h_j + w_i^T w_j + v^T d_j)
        '''
        a = (w * h_j).sum(dim=1)  # 普通的矩阵相乘
        b = torch.mm(w, k_j.unsqueeze(dim=1)).squeeze()  # batch
        c = torch.mm(d_j, self.v.unsqueeze(dim=1)).squeeze()  # batch
        return torch.sigmoid(a + b + c)
