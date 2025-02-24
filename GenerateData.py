import numpy as np
import matplotlib.pyplot as plt
from show_politope import *
from scipy.optimize import linprog


def generate_valid_H_representation(rng, d=3, m=10):
    """
    Generate a random non-empty, bounded polytope in H-representation (Ax <= b).

    Parameters:
    d (int): Dimension of the polytope.
    m (int): Number of half-spaces (inequalities).

    Returns:
    A (numpy.ndarray): Constraint matrix (m x d).
    b (numpy.ndarray): Constraint vector (m,).
    x0 (numpy.ndarray): A feasible interior point.
    """

    while True:  # Keep generating until a valid polytope is found
        # Step 1: Generate a feasible interior point
        x0 = rng.uniform(-1, 1, size=(d,))

        # Step 2: Generate A with normal vectors from a sphere (ensuring diverse directions)
        A = rng.normal(size=(m, d))
        A /= np.linalg.norm(A, axis=1, keepdims=True)  # Normalize rows to unit length

        # Step 3: Compute b ensuring feasibility (Ax0 <= b)
        b = A @ x0 + rng.uniform(0.1, 1.0, size=(m,))

        # Step 4: Check feasibility (Ax <= b must have a solution)
        res = linprog(np.zeros(d), A_ub=A, b_ub=b, method='highs')
        if res.success:  # If feasible, check boundedness
            
            # Step 5: Check boundedness by trying to maximize x in all directions
            bounded = True
            for i in range(d):
                obj = np.zeros(d)
                obj[i] = 1  # Maximize along one axis
                res = linprog(obj, A_ub=A, b_ub=b, bounds=[(None, None)] * d, method='highs')
                if res.success and res.status == 3:  # Unbounded solution detected
                    bounded = False
                    break
            
            if bounded:
                return A, b, x0  # Return only if it's feasible and bounded





def generate_inequalities(rng, m=4, r=3):
    #Ax <= b
    #m = numero di disuguaglianze
    #r = numero di dimensioni
    
    A_valid, b_valid, x0_valid = generate_valid_H_representation(rng, d=r, m=m)
    return A_valid, b_valid
    
    #x0 = np.zeros(r)
    x0 = np.full(r, 1)
    
    A = rng.uniform(-10, 10, (m, r))
    #b = rng.uniform(-10, 10, m)
    
    print("A", A)
    print("\n")
    b = A @ x0
    print("b", b)
    #b += rng.uniform(0.1, 1.0, size=(m,))
    print("b", b)
    #b += 10
    print("b", b)
    
    return A, b 

def show_politope(A, b):
    sol = np.array([0,0,0])

    print("sol", sol)
    plot_politope(A, b)
    pass
    

if __name__ == "__main__":
    seed = 2
    rng = np.random.default_rng(seed)
    
    A, b = generate_inequalities(rng, m=6, r=3)
    print(A)
    print(b)
    
    """A = np.array([
        [1, 0, 0],  # x <= 1
        [-1, 0, 0], # x >= -1
        [0, 1, 0],  # y <= 1
        [0, -1, 0], # y >= -1
        [0, 0, 1],  # z <= 1
        [0, 0, -1]  # z >= -1
        ])
    b = np.array([1, 1, 1, 1, 1, 1])"""
    
    show_politope(A, b)