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
import data


def main(**kwargs):
    config = DefaultConfig()
    config.parse(kwargs)
    if not torch.cuda.is_available():
        config.cuda = False
        config.device = None
        torch.manual_seed(config.seed)

    train_iter, val_iter, test_iter, config.vocab_size, vectors = data.load_data(config)

    config.print_config()


if __name__ == '__main__':
    fire.Fire()
