import numpy as np
import matplotlib.pyplot as plt
from show_politope import *
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull

def is_bounded(A, b):
    n = A.shape[1]
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1  # Massimizza una singola variabile alla volta
        res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
        if res.status == 3:  # Stato 3 indica soluzioni illimitate
            return False
    return True

def generate_politope(rng, m=4, r=3):
    #Ax <= b
    #m = numero di disuguaglianze
    #r = numero di dimensioni
    
    x0 = np.full(r, 1)
    A = rng.uniform(-10, 10, (m, r))
    b = A @ x0
    b += rng.uniform(1, 30, m)
    
    is_valid = is_bounded(A, b)
    
    return is_valid, A, b


def show_politope(A, b):
    plot_politope(A, b)


if __name__ == "__main__":
    seed = 5
    rng = np.random.default_rng(seed)
    
    is_valid, A, b = generate_politope(rng, m=6, r=3)
    #print("is_valid", is_valid)
    #print(A)
    #print(b)
    
    show_politope(A, b)