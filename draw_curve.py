# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-5 下午12:34
@ide     : PyCharm  
"""

import matplotlib.pyplot as plt
import numpy as np

avg_cost = []
train_acc = []
valid_acc = []
test_acc = []
with open('train_result.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'Avg Cost' in line:
            avg_cost.append(line.strip().split('Avg Cost: ')[1])
        if 'Training Accuracy' in line:
            train_acc.append(line.strip().split('Training Accuracy: ')[1])
        if 'Validation Accuracy' in line:
            valid_acc.append(line.strip().split('Validation Accuracy: ')[1])
        if 'Test Accuracy' in line:
            test_acc.append(line.strip().split('Test Accuracy: ')[1])

print('Min Average Cost: ' + min(avg_cost))
print('Max Train Accuracy: ' + max(train_acc))
print('Max Validation Accuracy: ' + max(valid_acc))
print('Max Test Accuracy: ' + max(test_acc))

# choose the best model with respect to the dev set
best_epoch = max(zip(range(800), valid_acc), key=lambda x: x[1])[0]
print('\nBest Model is in epoch ' + str(best_epoch))
print('Valid Acc of Model in epoch {}: {}'.format(best_epoch, valid_acc[best_epoch]))
print('Test Acc of the Best Model: ' + test_acc[best_epoch])

# use matplotlib draw curve
epochs = range(0, 800, 10)
plt.figure()
plt.plot(epochs, avg_cost[::10])
# plt.ylim((0, 1))
plt.show()
