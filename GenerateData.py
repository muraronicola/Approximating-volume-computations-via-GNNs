import numpy as np
import matplotlib.pyplot as plt
from show_politope import *
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
import polytope as pc

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


def load_data(filename):
    data = np.load(filename, allow_pickle=True)
    return data


def test_main():
    seed = 5
    rng = np.random.default_rng(seed)
    
    is_valid, A, b = generate_politope(rng, m=6, r=3)
    #print("is_valid", is_valid)
    #print(A)
    #print(b)
    
    """A = np.array([
        [1, 0, 0],  # x <= 1
        [-1, 0, 0], # x >= -1
        [0, 1, 0],  # y <= 1
        [0, -1, 0], # y >= -1
        [0, 0, 1],  # z <= 1
        [0, 0, -1]  # z >= -1
        ])
    b = np.array([1, 1, 1, 1, 1, 1])"""
    
    plot_politope(A, b)
    
    p = pc.Polytope(A, b)
    print("Volume:", p.volume)


def main():
    n_politopes = 200
    seed = 0
    rng = np.random.default_rng(seed)

    base_path = "./data/"
    data = []
    file = open(base_path + "politopes.txt", "w")   
    save_images = False 
    
    i = 0
    while i < n_politopes:
        is_valid, A, b = generate_politope(rng, m=6, r=3)
        if is_valid:
            status = plot_politope(A, b, save=save_images, show=False, filename=base_path + "politope_" + str(i) + ".png")
            
            if status:
                p = pc.Polytope(A, b)
                data.append([A, b, p.volume])
                i += 1
    
    data = np.array(data, dtype=object)
    np.save(base_path + "politopes.npy", data)
    file.write(str(data))
    

if __name__ == "__main__":
    #test_main()
    main()