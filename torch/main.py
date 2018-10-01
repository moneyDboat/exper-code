# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/10/1 20:15
# @Ide     : PyCharm
"""

import torch
import torch.nn.functional as F
from config import DefaultConfig
import fire


def main(**kwargs):
    args = DefaultConfig()
    args.parse(kwargs)
    if not torch.cuda.is_available():
        args.cuda = False
        args.device = None
        torch.manual_seed(args.seed)

    # 只考虑4个top aspect
    aspect2idx = {
        'general': 0,
        'price': 1,
        'transit-location': 2,
        'safety': 3
    }
