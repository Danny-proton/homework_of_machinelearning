from scipy import *

def loss(c, all_points):
    return sum(sum((c - all_points) ** 2, axis=1))
