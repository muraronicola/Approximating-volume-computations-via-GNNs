import numpy as np
import matplotlib.pyplot as plt
from show_politope import *
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
from pycvxset import Polytope, Ellipsoid, spread_points_on_a_unit_sphere
import os
from tqdm.contrib.itertools import product

def is_bounded(A, b):
    n = A.shape[1]
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1
        res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
        if not res.success:
            return False
        
    c = np.full(n, -1)
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    return res.success

def generate_polytope(rng, m=4, r=3):
    #Ax <= b
    #m = numero di disuguaglianze
    #r = numero di dimensioni
    
    x0 = np.full(r, 1)
    A = rng.uniform(-10, 10, (m, r))
    b = A @ x0
    b += rng.uniform(1, 30, m)
    
    is_valid = is_bounded(A, b)
    
    return is_valid, A, b


def generate_n_polytopes(n_polytopes, base_path="./data/", seed=0, m=4, r=3, save_images=True):
    rng = np.random.default_rng(seed)

    path = base_path + "m_" + str(m) + "_r_" + str(r) + "/"
    os.makedirs(path, exist_ok=True)
    
    data = []
    
    i = 0
    invalids = 0
    while i < n_polytopes:
        is_valid, A, b = generate_polytope(rng, m=m, r=r)
        
        if is_valid:
            polytope = Polytope(A=A, b=b)
            
            if r <= 3:
                plot_polytope(polytope, save=save_images, show=False, filename=path + "politope_" + str(i) + ".png")
            
            volume = polytope.volume()
            data.append([A, b, volume])
            i += 1
            invalids = 0
        else:
            invalids += 1
            if invalids > 100:
                return False
        
        if i == 10 and save_images: 
            save_images = False

    data = np.array(data, dtype=object)
    np.save(path + "polytopes.npy", data)
    
    return True


def load_data(filename):
    data = np.load(filename, allow_pickle=True)
    return data



def main():
    m_array = range(3, 30)
    r_array = range(2, 5)
    n_polytopes = 50
    seed = 0
    
    for r, m in product(r_array, m_array):
        status = generate_n_polytopes(n_polytopes, base_path="./data/", seed=seed, m=m, r=r)
        if not status:
            print("Error in generating polytopes with m =", m, "and r =", r)


if __name__ == "__main__":
    #test_main()
    main()
    
    





def test_main():
    seed = 5
    rng = np.random.default_rng(seed)
    
    is_valid, A, b = generate_polytope(rng, m=6, r=3)
    #print("is_valid", is_valid)
    #print(A)
    #print(b)
    
    A = np.array([
        [20, 0, 0],  # x <= 1
        [-1, 1, 0], # x >= -1
        [0, 1, 0],  # y <= 1
        [0, -1, 0], # y >= -1
        [0, 0, 1],  # z <= 1
        [0, 0, -1]  # z >= -1
        ])
    b = np.array([1, 1, 1, 1, 1, 1])
    
    
    politope = Polytope(A=A, b=b)
    status = plot_polytope(politope, save=False, show=False)
    
    p = pc.Polytope(A, b)
    print("Volume:", p.volume)
    print("Volume:", politope.volume())
    