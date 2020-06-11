#31917070 丁奕飞

from scipy import *
import pylab as pl  # pylab 模块是一款由python提供的可以绘制二维，三维数据的工具模块，其中包括了绘图软件包 matplotlib
from loss_function import loss
from updata_method import *



all_points = rand(2000, 2)

start = array([0, 1])  # 出发点
lr = 0.03  # 学习率
loop_max = 1000  # 最大迭代次数(防止死循环)
epsilon = 1e-6  # 设置阈值
xb = start
x = start
#GD训练过程
for i in range(loop_max):
    loss1 = loss(x, all_points)  # 梯度更新前的损失函数值
    xi = x - lr * GD(x, all_points)  # 梯度更新后的新的点
    lossi = loss(xi, all_points)  # 更新后的损失函数值
    if loss1 - lossi > epsilon:  # 更新前损失函数值减去更新后的差大于阈值，继续循环
        x = xi
        loss1 = lossi
    elif lossi - loss1 > epsilon:  # 更新后损失函数值减去更新前的差大于阈值，说明步长过大，需要调小
        lr = lr * 0.3
    else:
        break
    xb = vstack((xb, x))
#SGD训练过程
for i in range(loop_max):
    from random import choice

    r = choice(all_points)
    loss1 = loss(x, all_points)
    xi = x - lr * SGD(x, r)
    lossi = loss(xi, all_points)
    if loss1 - lossi > epsilon:
        x = xi
        loss1 = lossi
    elif lossi - loss1 > epsilon:
        lr = lr * 0.5
    else:
        break
    xb = vstack((xb, x))

c = x

pl.plot(all_points[:, 0], all_points[:, 1], 'g.')
pl.plot(xb[:, 0], xb[:, 1], 'r.')
pl.plot(xb[:, 0], xb[:, 1], 'k-')
pl.xlabel('c = (%.3f, %.3f)' % (c[0], c[1]))

pl.show()

print(c)
#
# x = array([0, 1])  # 出发点
#
# lr = 0.08  # 学习率
# xb = x
# loop_max = 10000  # 最大迭代次数(防止死循环)
# epsilon = 1e-8
#
# for i in range(loop_max):
#     from random import choice
#
#     r = choice(all_points)
#     loss1 = loss(x, all_points)
#     xi = x - lr * SGD(x, r)
#     lossi = loss(xi, all_points)
#     if loss1 - lossi > epsilon:
#         x = xi
#         loss1 = lossi
#     elif lossi - loss1 > epsilon:
#         lr = lr * 0.5
#     else:
#         break
#     xb = vstack((xb, x))
#
# c = x
#
# pl.plot(all_points[:, 0], all_points[:, 1], 'g.')
# pl.plot(xb[:, 0], xb[:, 1], 'r.')
# pl.plot(xb[:, 0], xb[:, 1], 'k-')
# pl.xlabel('c = (%.3f, %.3f)' % (c[0], c[1]))
#
# pl.show()
#
# print(c)