import numpy as np
from show_politope import *
from scipy.optimize import linprog
from pycvxset import Polytope
import os
import argparse
from tqdm import tqdm
import math


# Check if the polytope is bounded
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


# Generate a random polytope with m constraints and r dimensions, Ax <= b
def generate_polytope(rng, m=4, r=3):
    x0 = np.full(r, 1)
    A = rng.uniform(-10, 10, (m, r))
    b = A @ x0
    b += rng.uniform(1, 40, m)
    
    is_valid = is_bounded(A, b) #Check if the polytope is bounded
    
    return is_valid, A, b



# Check if all the constraints are useful for defining the polytope
def check_exact_polytope(original_volume, A, b, m): 
    exact = True
    
    i = 0
    while i < m and exact:
        A_i = np.delete(A, i, axis=0)
        b_i = np.delete(b, i)
        
        is_valid = is_bounded(A_i, b_i)
        
        if is_valid:
            p = Polytope(A=A_i, b=b_i)
            new_volume = p.volume()

            if new_volume < original_volume + original_volume * 0.05: #The volume must change at least 5% to be considered a different polytope
                exact = False
        
        i += 1
        
    return exact


# Generate n polytopes with m constraints and r dimensions, Ax <= b
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
    
    #In order to have a uniform distribution of the volumes, we need to discretize the volume space
    discretization_step = 5 #Size of volume in which there are going to be the same number of politopes
    volume_counter = np.zeros(math.floor(max_volume/discretization_step)) 
    target_volume = math.floor(n_polytopes/(max_volume/discretization_step))
    
    while i < n_polytopes:
        finite_volume = True
        
        if uniform:
            if (volume_counter == target_volume).all(): #If each slot of volume_counter is full, we have done
                break
        
        is_valid, A, b = generate_polytope(rng, m=m, r=r) #Get a random polytope
        
        if is_valid:
            polytope = Polytope(A=A, b=b)

            try:
                volume = polytope.volume() # Based on ConvexHull, it returns the volume of the polytope
            except:
                print("Error in volume calculation, not enough points")
                finite_volume = False
            
            if finite_volume:
                x = np.concatenate((A, b.reshape(-1, 1)), axis=1) #x is the matrix containing the H representation of the polytope
                
                if not only_exact: #If we want to save all the polytopes, not only the exact ones
                    data_x.append(x)
                    data_y.append(volume)
                
                if check_exact_polytope(volume, A, b, m): #Check if the polytope is defined by all the constraints
                    if (not uniform) or (uniform and volume < max_volume and volume_counter[round(volume/discretization_step) - 1] < target_volume): #If the generated polytope has a volume that we need in order to have a uniform distribution
                        
                        if r <= 3 and save_images:
                            plot_polytope(polytope, save=save_images, show=False, filename=path + "politope_" + str(i) + ".png")
                        
                        data_exact_politope_x.append(x)
                        data_exact_politope_y.append(volume)
                        volume_counter[round(volume/discretization_step) - 1] += 1 #Increment the counter of the volume for the specific slot (used in order to have a uniform distribution)
                        
                        i += 1
                        pbar.update(1)
            
        if i == 10 and save_images: 
            save_images = False


    #Save the data in a numpy array to disk
    data_exact_politope_x = np.array(data_exact_politope_x, dtype=object)
    data_exact_politope_y = np.array(data_exact_politope_y, dtype=object)
    np.save(path + "exact_politopes_x.npy", data_exact_politope_x)
    np.save(path + "exact_politopes_y.npy", data_exact_politope_y)
    
    #If we want to save all the polytopes, not only the exact ones
    if not only_exact:
        data_x = np.array(data_x, dtype=object)
        data_y = np.array(data_y, dtype=object)
        
        np.save(path + "all_polytopes_x.npy", data_x)
        np.save(path + "all_polytopes_y.npy", data_y) 
    
    #Normalize the X data and save it
    if normalize:
        for i in range(len(data_exact_politope_x)):
            data_exact_politope_x[i] = data_exact_politope_x[i] / np.linalg.norm(data_exact_politope_x[i])
            
        np.save(path + "exact_politopes_x_normalized.npy", data_exact_politope_x)


#Main logic of the application
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dimensions", type=int, default=0, help="number of dimensions")
    parser.add_argument("--constraints", type=int, default=0, help="number of constraints")
    parser.add_argument("--number_polytopes", type=int, default=10000, help="number of polytopes to generate")
    parser.add_argument("--uniform_distribution", type=bool, default=True, help="uniform distribution?")
    parser.add_argument("--max_volume", type=int, default=500, help="max volume (only for uniform distribution)")
    parser.add_argument("--normalize", type=bool, default=True, help="Normalization of the final data")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random number generator")
    parser.add_argument("--only_exact", type=bool, default=True, help="Only exact polytopes? (defined by all the constraints)")

    args = parser.parse_args()

    r = args.dimensions
    m = args.constraints
    n_polytopes = args.number_polytopes
    uniform = args.uniform_distribution
    max_volume = args.max_volume
    normalize = args.normalize
    seed = args.seed
    only_exact = args.only_exact

    if m > r:
        print("\nGenerating polytopes with m =", m, "and r =", r)
        generate_n_polytopes(n_polytopes, base_path="./data/", seed=seed, m=m, r=r, only_exact=only_exact, uniform=uniform, max_volume=max_volume, normalize=normalize)
    else:
        print("The number of constraints must be greater than the number of dimensions")


if __name__ == "__main__":
    main()

