# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/10/1 20:17
# @Ide     : PyCharm
"""


class DefaultConfig(object):
    # 所有可设置的参数
    learning_rate = 0.05
    max_grad_norm = 5.0
    evaluation_interval = 1
    batch_size = 128
    epochs = 50
    sentence_len = 50
    task = 'Sentihood'
    seed = 777
    data_dir = '../data/sentihood'
    opt = 'ftrl'
    embedding_path = '../data/glove.6B.300d.txt'
    embedding_update = False
    case_folding = True

    n_keys = 7
    n_tied = 2
    entnet_input_keep_prob = 0.8
    entnet_output_keep_prob = 1.0
    entnet_state_keep_prob = 1.0
    final_layer_keep_prob = 0.8
    l2_final_layer = 1e-3

    # mine
    cuda = True
    device = 0

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception("Warning: config has not attribute {}".format(k))
            setattr(self, k, v)

    def print_config(self):
        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'parse' and k != 'print_config':
                print('  {} : {}'.format(k, getattr(self, k)))