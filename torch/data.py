# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/10/1 21:07
# @Ide     : PyCharm
"""

from torchtext import data
from torchtext.vocab import Vectors
from tqdm import tqdm


# 定义Dataset
class SentiDataset(data.Dataset):
    name = 'Sentihood'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, aspect_field, label_field, test=False, **kwargs):
        fields = [('text', text_field), ('aspect', aspect_field), ('label', label_field)]
        examples = []
        print('read data from {}'.format(path))

        if test:
            # 如果为测试集，则不加载label
