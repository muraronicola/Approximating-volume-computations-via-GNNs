import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from pycvxset import Polytope, Ellipsoid, spread_points_on_a_unit_sphere

# Define the H-representation (Ax <= b)

def plot_politope(A, b):
        """        A = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        )
        b =  np.array([2, 3, 1, 2, 3, 1])"""
        
        print("A", A)
        print("b", b)
        P_hrep_3D = Polytope(A=A, b=b)

        ax, _, _ = P_hrep_3D.plot(patch_args={"label": "P_hrep_3D"})
        ax.legend()
        ax.set_title("Plotting P_hrep_3D")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        plt.show()
        
        return 
        """A = np.array([
        [1, 0, 0],  # x <= 1
        [-1, 0, 0], # x >= -1
        [0, 1, 0],  # y <= 1
        [0, -1, 0], # y >= -1
        [0, 0, 1],  # z <= 1
        [0, 0, -1]  # z >= -1
        ])
        b = np.array([1, 1, 1, 1, 1, 1])"""

        # Find an interior point for the intersection algorithm
        
        A = A.astype(np.float64)
        b = b.astype(np.float64)
        
        n_runs = 10
        
        solutions = []
        
        for i in range(n_runs):
                new_A = np.copy(A)
                new_b = np.copy(b)
                
                #Add random noise to the constraints
                new_A += np.random.normal(0, 0.1, new_A.shape)
                new_b += np.random.normal(0, 0.1, new_b.shape)
        
                res = linprog(np.zeros((new_A.shape[1],)), A_ub=new_A, b_ub=new_b, bounds=(None, None), method='highs-ipm')
                interior_point = np.array([res.x[0], res.x[1], res.x[2]])
                solutions.append(interior_point)
        
        solutions = np.array(solutions)
        sol = np.mean(solutions, axis=0)
        print("final sol", sol)

        # Compute the polytope vertices using HalfspaceIntersection
        print("input", np.hstack((A, -b[:, None])))
        hs = HalfspaceIntersection(np.hstack((A, -b[:, None])), sol)
        vertices = hs.intersections

        # Compute the convex hull for visualization
        hull = ConvexHull(vertices)
        outer_vertices = vertices[hull.vertices]

        # Plot the polytope
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=hull.simplices, color="cyan", alpha=0.6, edgecolor="k")


        print("vertices", vertices)
        print("outer_vertices", outer_vertices)


        # Labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
