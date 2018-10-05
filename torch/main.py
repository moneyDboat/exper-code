# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/10/1 20:15
# @Ide     : PyCharm
"""

import torch
import torch.nn.functional as F
from config import DefaultConfig
from model import EntNet
import fire
import data
import os


def main(**kwargs):
    config = DefaultConfig()
    config.parse(kwargs)
    if not torch.cuda.is_available():
        config.cuda = False
        config.device = None
        torch.manual_seed(config.seed)

    train_iter, val_iter, test_iter, config.vocab_size, config.target_vocab_size, config.aspect_vocab_size, \
        text_vectors, target_vectors, aspect_vectors = data.load_data(config)
    # 需要进一步处理样本不均衡

    config.print_config()

    # # init model
    # model = EntNet(config, text_vectors, target_vectors, aspect_vectors)
    # print(model)
    #
    # # 模型保存位置
    # if not os.path.exists(config.save_dir):
    #     os.mkdir(config.save_dir)
    # save_path = os.path.join(config.save_dir, 'entnet_{}.pth'.format(config.id))
    #
    # if config.cuda:
    #     torch.cuda.set_device(config.device)
    #     torch.cuda.manual_seed(config.seed)  # set random seed for gpu
    #     model.cuda()
    #
    # # 目标函数和优化器
    # criterion = F.cross_entropy
    # optimizer = None

    # 开始训练
    for i in range(config.max_epoch):
        total_loss = 0.0
        correct = 0
        total = 0

        # model.train()

        for idx, batch in enumerate(train_iter):
            text, target, aspect, label = batch.text, batch.target, batch.aspect, batch.label
            if config.cuda:
                text, target, aspect, label = text.cuda(), target.cuda(), aspect.cuda(), label.cuda()

            optimizer.zero_grad()
            pred = model(text, target, aspect)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            # 更新统计指标
            pass

        # 计算验证集上的分数，并相应调整学习率
        pass

    # 保存最终的训练模型
    print('Best final model saved in {}'.format(save_path))

    # 在测试集上计算结果


if __name__ == '__main__':
    fire.Fire()
