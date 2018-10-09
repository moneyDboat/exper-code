# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/10/1 20:15
# @Ide     : PyCharm
"""

import torch
import torch.nn.functional as F
from config import DefaultConfig
import models
import fire
import data
import os
import time
from util import Visualizer


def main(**kwargs):
    start_time = time.time()

    config = DefaultConfig()
    config.parse(kwargs)

    vis = Visualizer(config.env)

    if not torch.cuda.is_available():
        config.cuda = False
        config.device = None
        torch.manual_seed(config.seed)

    train_iter, val_iter, test_iter, config.vocab_size, config.target_vocab_size, config.aspect_vocab_size, \
    text_vectors, target_vectors, aspect_vectors = data.load_data(config)
    # 需要进一步处理样本不均衡

    config.print_config()

    # init model
    model = getattr(models, config.model)(config, text_vectors, target_vectors, aspect_vectors)
    print(model)

    # 模型保存位置
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    tmp_save_path = os.path.join(config.save_dir, 'entnet_{}.pth'.format(config.id))

    if config.cuda:
        torch.cuda.set_device(config.device)
        torch.cuda.manual_seed(config.seed)  # set random seed for gpu
        model.cuda()

    # 目标函数和优化器
    criterion = F.cross_entropy
    lr1, lr2 = config.lr1, config.lr2
    optimizer = model.get_optimizer(lr1, lr2)

    global best_acc
    best_acc = 0.0

    # 开始训练
    for i in range(config.max_epoch):
        total_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for idx, batch in enumerate(train_iter):
            text, target, aspect, label = batch.text, batch.target, batch.aspect, batch.label
            if config.cuda:
                text, target, aspect, label = text.cuda(), target.cuda(), aspect.cuda(), label.cuda()

            optimizer.zero_grad()
            pred = model(text, target, aspect)
            loss = criterion(pred, label)
            loss.backward(retain_graph=True)
            optimizer.step()

            # 更新统计指标
            total_loss += loss.item()
            predicted = pred.max(dim=1)[1]
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

        # 每个batch之后计算测试集上的准确率
        print('[Epoch {}] loss: {:.5f} | Acc: {:.3f}%({}/{})'.format(i + 1, total_loss,
                                                                     100. * correct / total, correct, total))
        vis.plot('loss', total_loss)

        # 每5个epoch计算验证集上的准确率，并相应调整学习率
        if i % 5 == 4:
            acc, acc_n, val_n = val(model, val_iter, config)
            vis.plot('val_acc', acc)
            print('Epoch {} Val Acc: {:.3f}%({}/{})'.format(i + 1, acc, acc_n, val_n))
            # 100个epoch之后模型接近收敛，此时开始调整学习率
            # 因为数据集偏小，100个epoch之前虽然整体呈下降趋势，但会有小幅度波动，此时调整学习率可能会影响模型收敛
            if i > 100:
                if acc >= best_acc:
                    best_acc = acc
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'config': config
                    }
                    torch.save(checkpoint, tmp_save_path)
                # if acc < best_acc:
                #     model.load_state_dict(torch.load(tmp_save_path)['state_dict'])
                    # lr1 *= config.lr_delay
                    # optimizer = model.get_optimizer(lr1, lr2)
                    # print('## load previous best model: {:.3f}%'.format(best_acc))
                    # print('## set model lr1 to {}'.format(lr1))
                    # if lr1 < config.min_lr:
                    #     print('## training over, best f1 acc : {:.3f}'.format(best_acc))
                    #     break

            # 计算测试集上分数(准确率)
            test_acc, test_acc_n, test_n = val(model, test_iter, config)
            vis.plot('test_acc', test_acc)
            print('Epoch {} Test Acc: {:.3f}%({}/{})\n'.format(i + 1, test_acc, test_acc_n, test_n))

    # 加载训练过程中保存的验证集最佳模型
    # 计算最终训练模型的测试集准确率
    model.load_state_dict(torch.load(tmp_save_path)['state_dict'])
    print('Load tmp best model from {}'.format(tmp_save_path))
    test_acc, test_acc_n, test_n = val(model, test_iter, config)
    print('Finally Test Acc: {:.3f}%({}/{})'.format(test_acc, test_acc_n, test_n))
    # print('Best final model saved in {}'.format('{:.3f}_{}'.format(test_acc, tmp_save_path)))

    print('Final cost time : {}s'.format(time.time() - start_time))


# 计算模型在验证集/测试集上的结果(准确率)
def val(model, datasets, config):
    # 将模型设为验证模式
    model.eval()

    acc_n = 0
    val_n = 0

    with torch.no_grad():
        for batch in datasets:
            text, target, aspect, label = batch.text, batch.target, batch.aspect, batch.label
            if config.cuda:
                text, target, aspect, label = text.cuda(), target.cuda(), aspect.cuda(), label.cuda()

            pred = model(text, target, aspect)
            predicted = pred.max(dim=1)[1]
            acc_n += (predicted == label).sum().item()
            val_n += label.size(0)

    acc = 100. * acc_n / val_n
    return acc, acc_n, val_n


if __name__ == '__main__':
    fire.Fire()
