import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from pycvxset import Polytope, Ellipsoid, spread_points_on_a_unit_sphere

def plot_politope(A, b, save=False, filename=None):
    P_hrep_3D = Polytope(A=A, b=b)

    ax, _, _ = P_hrep_3D.plot(patch_args={"label": "P_hrep_3D"})
    ax.legend()
    ax.set_title("Plotting P_hrep_3D")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    if not save:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
