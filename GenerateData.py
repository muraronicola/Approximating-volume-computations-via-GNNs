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


def check_exact_polytope(original_volume, A, b, m): #Check if all the constraints are useful
    for i in range(m):
        A_i = np.delete(A, i, axis=0)
        b_i = np.delete(b, i)
        
        is_valid = is_bounded(A_i, b_i)
        if is_valid:
            p = Polytope(A=A_i, b=b_i)
            new_volume = p.volume()

            if new_volume < original_volume + original_volume * 0.05: #The constraint is that the volume must change at least 5% to be considered a different polytope
                return False
        
    return True

def generate_n_polytopes(n_polytopes, base_path="./data/", seed=0, m=4, r=3, save_images=True, only_exact=False):
    rng = np.random.default_rng(seed)

    path = base_path + "m_" + str(m) + "_r_" + str(r) + "/"
    os.makedirs(path, exist_ok=True)
    
    data_x = []
    data_y = []
    data_exact_politope_x = []
    data_exact_politope_y = []
    
    i = 0
    while i < n_polytopes:
        is_valid, A, b = generate_polytope(rng, m=m, r=r)
        
        if is_valid:
            polytope = Polytope(A=A, b=b)

            try:
                volume = polytope.volume()
            except:
                print("Error in volume calculation, not enough points")
                continue
            
            x = np.concatenate((A, b.reshape(-1, 1)), axis=1)
            if not only_exact:
                data_x.append(x)
                data_y.append(volume)
            
            if check_exact_polytope(volume, A, b, m):
                if r <= 3:
                    plot_polytope(polytope, save=save_images, show=False, filename=path + "politope_" + str(i) + ".png")
                
                data_exact_politope_x.append(x)
                data_exact_politope_y.append(volume)
                i += 1
        
        if i == 10 and save_images: 
            save_images = False

    if not only_exact:
        data_x = np.array(data_x, dtype=object)
        data_y = np.array(data_y, dtype=object)
        
    data_exact_politope_x = np.array(data_exact_politope_x, dtype=object)
    data_exact_politope_y = np.array(data_exact_politope_y, dtype=object)
    
    if not only_exact:
        np.save(path + "all_polytopes_x.npy", data_x)
        np.save(path + "all_polytopes_y.npy", data_y) 
        
    np.save(path + "exact_politopes_x.npy", data_exact_politope_x)
    np.save(path + "exact_politopes_y.npy", data_exact_politope_y)
    

def load_data(filename):
    data = np.load(filename, allow_pickle=True)
    return data


def main():
    m_array = range(3, 8) #[4, 5, 6] #range(3, 8)
    r_array = range(3, 5) #[3, 4] #range(4, 5)
    n_polytopes = 100 #10000 #50
    seed = 0
    
    for r, m in product(r_array, m_array):
        if m > r:
            generate_n_polytopes(n_polytopes, base_path="./data/", seed=seed, m=m, r=r, only_exact=False)
        else:
            print("m must be greater than r")

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
    