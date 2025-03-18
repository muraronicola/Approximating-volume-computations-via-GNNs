import numpy as np
import matplotlib.pyplot as plt
from show_politope import *
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
from pycvxset import Polytope, Ellipsoid, spread_points_on_a_unit_sphere
import os
import argparse
from tqdm import tqdm
import math


A_1= np.array([[1,1], [1,0], [-2,-1]])
b_1 = np.array([5,3,-4])

A_2= np.array([[1,1], [1,0], [2,1]])
b_2 = np.array([5,3,4])

n = A_1.shape[1]

print("Maximizz A_1")
c = np.full(n, -1)
res = linprog(c, A_ub=A_1, b_ub=b_1, bounds=(None, None))
print(res)


print("Minimize A_1")
c = np.full(n, 1)
res = linprog(c, A_ub=A_1, b_ub=b_1, bounds=(None, None))
print(res)

print("--------------------")

print("Maximize A_2")
c = np.full(n, -1)
res = linprog(c, A_ub=A_2, b_ub=b_2, bounds=(None, None))
print(res)


print("Minimize A_2")
c = np.full(n, 1)
res = linprog(c, A_ub=A_2, b_ub=b_2, bounds=(None, None))
print(res)
