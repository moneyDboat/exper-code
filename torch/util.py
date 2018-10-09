# -*- coding: utf-8 -*-

"""
# @Author  : captain
# @Time    : 2018/10/1 20:16
# @Ide     : PyCharm
"""
import visdom
import numpy as np
import time


class Visualizer():
    '''
    封装了visdom的基本操作，但是可以通过self.vis.function调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横坐标
        # 保存('loss', 23)，即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        for k, v in d.iteritems():
            self.plot(k, v)

    def plot(self, name, y):
        # self.plot('loss', 1.00)

        x = self.index.get(name, 0)
        self.vis.line(X=np.array([x]), Y=np.array([y]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append')
        self.index[name] = x + 1

    def log(self, info, win='log_txt'):
        # self.log({'loss':1, 'lr':0.0001})
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, item):
        return getattr(self.vis, item)
