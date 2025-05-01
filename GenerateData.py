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

def is_bounded(A, b):
    ok = True
    
    n = A.shape[1]
    c = np.full(n, -1)
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    if not res.success:
        ok =  False
    
    c = np.full(n, 1)
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    if not res.success:
        ok = False
    
    return ok


def generate_polytope(rng, m=4, r=3):
    #Ax <= b
    #m = numero di disuguaglianze
    #r = numero di dimensioni
    
    x0 = np.full(r, 1)
    A = rng.uniform(-10, 10, (m, r))
    b = A @ x0
    b += rng.uniform(1, 40, m) #Era 1, 30, m
    
    is_valid = is_bounded(A, b)
    
    return is_valid, A, b


def check_exact_polytope(original_volume, A, b, m): #Check if all the constraints are useful
    exact = True
    
    i = 0
    while i < m and exact:
        A_i = np.delete(A, i, axis=0)
        b_i = np.delete(b, i)
        
        is_valid = is_bounded(A_i, b_i)
        
        if is_valid:
            p = Polytope(A=A_i, b=b_i)
            new_volume = p.volume()

            if new_volume < original_volume + original_volume * 0.05: #The constraint is that the volume must change at least 5% to be considered a different polytope
                exact = False
        
        i += 1
        
    return exact


def generate_n_polytopes(n_polytopes, base_path="./data/", seed=0, m=4, r=3, save_images=True, only_exact=False, uniform=True, max_volume=500, normalize=True):
    rng = np.random.default_rng(seed)

    path = base_path + "m_" + str(m) + "_r_" + str(r) + "/"
    os.makedirs(path, exist_ok=True)
    
    data_x = []
    data_y = []
    data_exact_politope_x = []
    data_exact_politope_y = []
    

    pbar = tqdm(total=n_polytopes)
    i = 0
    discretization_step = 5
    
    #In order to have a uniform distribution of the volumes, we need to discretize the volume space
    volume_counter = np.zeros(math.floor(max_volume/discretization_step)) 
    target_volume = math.floor(n_polytopes/(max_volume/discretization_step))
    
    while i < n_polytopes:
        finite_volume = True
        
        if uniform:
            if (volume_counter == target_volume).all():
                break
        
        is_valid, A, b = generate_polytope(rng, m=m, r=r)
        
        if is_valid:
            polytope = Polytope(A=A, b=b)

            try:
                volume = polytope.volume() # Based on ConvexHull
            except:
                print("Error in volume calculation, not enough points")
                finite_volume = False
            
            if finite_volume: 
                x = np.concatenate((A, b.reshape(-1, 1)), axis=1)
                
                if not only_exact:
                    data_x.append(x)
                    data_y.append(volume)
                
                if check_exact_polytope(volume, A, b, m):
                    if (not uniform) or (uniform and volume < max_volume and volume_counter[round(volume/discretization_step) - 1] < target_volume):
                        
                        if r <= 3 and save_images:
                            plot_polytope(polytope, save=save_images, show=False, filename=path + "politope_" + str(i) + ".png")
                        
                        data_exact_politope_x.append(x)
                        data_exact_politope_y.append(volume)
                        volume_counter[round(volume/discretization_step) - 1] += 1
                        
                        i += 1
                        pbar.update(1)
            
        if i == 10 and save_images: 
            save_images = False


    data_exact_politope_x = np.array(data_exact_politope_x, dtype=object)
    data_exact_politope_y = np.array(data_exact_politope_y, dtype=object)
    
    np.save(path + "exact_politopes_x.npy", data_exact_politope_x)
    np.save(path + "exact_politopes_y.npy", data_exact_politope_y)
    
    
    if not only_exact:
        data_x = np.array(data_x, dtype=object)
        data_y = np.array(data_y, dtype=object)
        
        np.save(path + "all_polytopes_x.npy", data_x)
        np.save(path + "all_polytopes_y.npy", data_y) 
    
    
    if normalize:
        for i in range(len(data_exact_politope_x)):
            data_exact_politope_x[i] = data_exact_politope_x[i] / np.linalg.norm(data_exact_politope_x[i])
            
        np.save(path + "exact_politopes_x_normalized.npy", data_exact_politope_x)
    

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", type=int, default=0, help="number of dimensions")
    parser.add_argument("-m", type=int, default=0, help="number of constraints")
    parser.add_argument("-n", type=int, default=10000, help="number of polytopes to generate")
    parser.add_argument("-u", type=bool, default=True, help="uniform distribution?")
    parser.add_argument("-max_v", type=int, default=500, help="max volume (only for uniform distribution)")
    parser.add_argument("--normalize", type=bool, default=True, help="Normalization of the final data")

    args = parser.parse_args()

    r = args.r
    m = args.m
    n_polytopes = args.n
    uniform = args.u
    max_volume = args.max_v
    normalize = args.normalize


    seed = 0

    if m > r:
        print("\nGenerating polytopes with m =", m, "and r =", r)
        generate_n_polytopes(n_polytopes, base_path="./data/", seed=seed, m=m, r=r, only_exact=True, uniform=uniform, max_volume=max_volume, normalize=normalize)
    else:
        print("m must be greater than r")

if __name__ == "__main__":
    main()

