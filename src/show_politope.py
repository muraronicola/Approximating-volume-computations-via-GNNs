import matplotlib.pyplot as plt

#Function to plot a polytope using matplotlib (max in 2D or 3D)
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
