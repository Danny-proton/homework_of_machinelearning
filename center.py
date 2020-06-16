#31917070 丁奕飞

from scipy import *
import pylab as pl  # pylab 模块是一款由python提供的可以绘制二维，三维数据的工具模块，其中包括了绘图软件包 matplotlib
from loss_function import loss
from updata_method import *
from random import choice


total_points=2000
all_points = rand(total_points, 2)
start = array([0, 1])  # 出发点
lr = 0.03  # 学习率
loop_max = 1000  # 最大迭代次数(防止死循环)
epsilon = 1e-6  # 设置阈值
xb = start
x = start

#GD训练过程
for i in range(loop_max):
    origin_loss = loss(x, all_points)  # 梯度更新前的损失函数值
    new_x = x - lr * GD(x, all_points)  # 梯度更新后的新的点
    new_loss = loss(new_x, all_points)  # 更新后的损失函数值
    if origin_loss - new_loss > epsilon:  # 更新前损失函数值减去更新后的差大于阈值，继续循环
        x = new_x
        origin_loss = new_loss
    elif new_loss - origin_loss > epsilon:  # 更新后损失函数值减去更新前的差大于阈值，说明步长过大，需要调小
        lr = lr * 0.3
    else:
        break
    xb = vstack((xb, x))

c = x

pl.plot(all_points[:, 0], all_points[:, 1], 'g.')
pl.plot(xb[:, 0], xb[:, 1], 'r.')
pl.plot(xb[:, 0], xb[:, 1], 'k-')
pl.xlabel('GD c = (%.3f, %.3f)' % (c[0], c[1]))

pl.show()

print(c)

#SGD训练过程


start = array([0, 1])  # 出发点
lr = 0.08  # 学习率
loop_max = 1000  # 最大迭代次数(防止死循环)
epsilon = 1e-8  # 设置阈值
xs = start
x = start

for i in range(loop_max):
    r = choice(all_points)
    origin_loss = loss(x, all_points)
    new_x = x - lr * SGD(x, r)
    # print(new_x,"---------------------------",new_x.shape,all_points.shape)
    new_loss = loss(new_x, all_points)
    if origin_loss - new_loss > epsilon:
        x = new_x
        origin_loss = new_loss
    elif new_loss - origin_loss > epsilon:
        lr = lr * 0.5
    else:
        break
    xs = vstack((xs, x))

c = x

pl.plot(all_points[:, 0], all_points[:, 1], 'g.')
pl.plot(xs[:, 0], xs[:, 1], 'b.')
pl.plot(xs[:, 0], xs[:, 1], 'k-')
pl.xlabel('SGD c = (%.3f, %.3f)' % (c[0], c[1]))

pl.show()

print(c)

# Mini-Batch GD
batch_size = 50  # 每次训练使用batch_size个点
start = array([0, 1])  # 出发点
lr = 0.08  # 学习率
loop_max = 1000  # 最大迭代次数(防止死循环)
epsilon = 1e-8  # 设置阈值
xm = start
x = start

total_batch = total_points//batch_size # 每个epoch迭代2000个点所需次数
for i in range(loop_max):
    begin = (i % total_batch) * batch_size
    end = begin + batch_size
    points = all_points[begin:end]
    # print(x, "---------------------------", x.shape, all_points.shape)
    origin_loss = loss(x,all_points)
    grad= lr * mini_batch_GD(x, points)
    # print(grad,grad.shape)
    new_x = x - grad
    # print(new_x, "---------------------------",new_x.shape,all_points.shape)
    new_loss = loss(new_x,all_points)
    if origin_loss - new_loss > epsilon:  # 更新前损失函数值减去更新后的差大于阈值，继续循环
        x = new_x
        origin_loss = new_loss
    elif new_loss - origin_loss > epsilon:  # 更新后损失函数值减去更新前的差大于阈值，说明步长过大，需要调小
        # print("ier %d: lr = %f, loss = %f" % (i, lr, loss))
        lr = lr * 0.8
    else:
        break
    xm = vstack((xm, x))
c = x
# true_xm = float(sum(all_points[:,0])/total_points)
# true_y = float(sum(all_points[:,1])/total_points)


pl.plot(all_points[:, 0], all_points[:, 1], 'g.')
pl.plot(xm[:, 0], xm[:, 1], 'y.')
pl.plot(xm[:, 0], xm[:, 1], 'k-')
pl.xlabel('mini batch c = (%.3f, %.3f)' % (c[0], c[1]))

pl.show()
print(c)




c = x

pl.plot(all_points[:, 0], all_points[:, 1], 'g.')
pl.plot(xs[:, 0], xs[:, 1], 'b.')
pl.plot(xs[:, 0], xs[:, 1], 'k-')
pl.plot(xb[:, 0], xb[:, 1], 'r.')
pl.plot(xb[:, 0], xb[:, 1], 'k-')
pl.plot(xm[:, 0], xm[:, 1], 'y.')
pl.plot(xm[:, 0], xm[:, 1], 'k-')
pl.xlabel('SGD vs. mini-batch GD vs. GD c = (%.3f, %.3f)' % (c[0], c[1]))

pl.show()

print(c)