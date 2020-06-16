from scipy import *

def loss(c, all_points):
    loss = sum((c - all_points) ** 2, axis=1) ** 0.5
    return sum(loss)
