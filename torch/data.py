# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/10/1 21:07
# @Ide     : PyCharm
"""

from torchtext import data
from torchtext import vocab
from tqdm import tqdm
from config import DefaultConfig
from torch.nn import init
import json


# 定义Dataset
class SentiDataset(data.Dataset):
    name = 'Sentihood'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, target_field, aspect_field, label_field, **kwargs):
        # 暂时只做二分类，后面需要加入三分类的数据预处理
        fields = [('text', text_field), ('target', target_field), ('aspect', aspect_field), ('label', label_field)]
        examples = []
        print('read data from {}'.format(path))

        # 只考虑4个top aspect
        # to index
        aspect2idx = {
            'general': 0,
            'price': 1,
            'transit-location': 2,
            'safety': 3
        }
        senti2idx = {
            'Positive': 0,
            'Negative': 1
        }

        # 从json中加载数据
        with open(path) as f:
            json_data = json.load(f)
            for json_item in json_data:
                # 全部转换成小写
                text = json_item['text'].lower()
                for opinion in json_item['opinions']:
                    sentiment = opinion['sentiment']
                    aspect = opinion['aspect']
                    target = opinion['target_entity']
                    if aspect in aspect2idx.keys():
                        examples.append(
                            data.Example.fromlist([text, target, aspect, senti2idx[sentiment]], fields))

        print('len of data in {} is : {}'.format(path, len(examples)))
        super(SentiDataset, self).__init__(examples, fields, **kwargs)


def load_data(config):
    TEXT = data.Field(sequential=True, fix_length=config.max_text_len)
    TARGET = data.Field(sequential=False, use_vocab=True)  # 'Location1' 或者 'Location2'
    ASPECT = data.Field(sequential=False, use_vocab=True)
    LABEL = data.Field(sequential=False, use_vocab=False)

    train_path = config.data_dir + 'sentihood-train.json'
    val_path = config.data_dir + 'sentihood-dev.json'
    test_path = config.data_dir + 'sentihood-test.json'

    # 数据预处理
    train = SentiDataset(train_path, text_field=TEXT, target_field=TARGET, aspect_field=ASPECT, label_field=LABEL)
    val = SentiDataset(val_path, text_field=TEXT, target_field=TARGET, aspect_field=ASPECT, label_field=LABEL)
    test = SentiDataset(test_path, text_field=TEXT, target_field=TARGET, aspect_field=ASPECT, label_field=LABEL)

    # 加载glove预训练词向量
    # 直接vocab.Glove，Glove类会自动通过url下载
    vectors = vocab.GloVe(name=config.embedding_name, dim=300)
    # vectors = Vectors(name=config.embedding_path)
    print('load glove {} vectors from url'.format(config.embedding_name))
    vectors.unk_init = init.xavier_uniform_  # 没有命中的token的初始化方式

    # 构建Vocab(target和aspect也需要单独构建)
    # 这样单独构建词向量其实是有问题的，aspect的词汇和text有一定关联，这样单独构建失去了这样的关联
    print('building vocabulary......')
    TEXT.build_vocab(train, val, test, min_freq=5, vectors=vectors)
    TARGET.build_vocab(train, val, test, vectors=vectors)
    ASPECT.build_vocab(train, val, test, vectors=vectors)

    # 构建Iterator
    train_iter = data.BucketIterator(dataset=train, batch_size=config.batch_size, shuffle=True, sort_within_batch=False,
                                     repeat=False, device=config.device)
    val_iter = data.Iterator(dataset=val, batch_size=config.batch_size, shuffle=False, sort=False, repeat=False,
                             device=config.device)
    test_iter = data.Iterator(dataset=test, batch_size=config.batch_size, shuffle=False, sort=False, repeat=False,
                              device=config.device)

    return train_iter, val_iter, test_iter, len(TEXT.vocab), len(TARGET.vocab), len(ASPECT.vocab), \
           TEXT.vocab.vectors, TARGET.vocab.vectors, ASPECT.vocab.vectors
