import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from show_politope import *

def generate_inequalities(rng, m=4, r=3):
    #Ax <= b
    #m = numero di disuguaglianze
    #r = numero di dimensioni
    
    A = rng.uniform(-10, 10, (m, r))
    b = rng.uniform(-10, 10, m)
    
    return A, b 

def show_politope(A, b):
    sol = np.linalg.solve(A, b) 
    plot_politope(A, b, sol)
    pass
    

if __name__ == "__main__":
    seed = 1
    rng = np.random.default_rng(seed)
    
    A, b = generate_inequalities(rng, m=3, r=3)
    print(A)
    print(b)
    
    show_politope(A, b)