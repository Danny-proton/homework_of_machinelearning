from scipy import *

def GD(c, all_points):
    dx = sum((c[0] - all_points[:, 0]) / sum((c - all_points) ** 2, axis=1) ** 0.5)  # 求x偏导数
    dy = sum((c[1] - all_points[:, 1]) / sum((c - all_points) ** 2, axis=1) ** 0.5)  # 求y偏导数
    # return normalization(dx,dy)  # 得到梯度向量
    return array([dx, dy])

def SGD(c, r):
	dx = (c[0] - r[0]) / sum((c - r) ** 2) ** 0.5 #求x偏导数
	dy = (c[1] - r[1]) / sum((c - r) ** 2) ** 0.5  #求y偏导数
    # return normalization(dx,dy)  # 得到梯度向量
	return array([dx, dy])#得到梯度向量

def normalization(dx,dy):
    s = (dx ** 2 + dy ** 2) ** 0.5
    dx = dx / s
    dy = dy / s
    return array([dx, dy])