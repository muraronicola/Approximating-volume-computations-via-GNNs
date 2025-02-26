import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from pycvxset import Polytope, Ellipsoid, spread_points_on_a_unit_sphere

def plot_polytope(polytope, save=False, show=True, filename=None):
    try:
        
        ax, _, _ = polytope.plot(patch_args={"label": "politope"})
        ax.legend()
        ax.set_title("Plotting politope")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        if polytope.dim == 3:
            ax.set_zlabel("z")
        
        if show:
            plt.show()
            
        if save and filename is not None:
            plt.savefig(filename)
        
        plt.close()
        return True
    
    except Exception as e:
        return False
