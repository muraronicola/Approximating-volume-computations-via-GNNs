import matplotlib.pyplot as plt 
from matplotlib import cm 
import numpy as np


def plot_politope(A, b, sol):
    w1 = b[0]
    w2 = b[1]
    w3 = b[2]
    
    x1 = A[0][0]
    x2 = A[1][0]
    x3 = A[2][0]
    
    y1 = A[0][1]
    y2 = A[1][1]
    y3 = A[2][1]
    
    z1 = A[0][2]
    z2 = A[1][2]
    z3 = A[2][2]
    
    # Returns number spaces evenly w.r.t 
    # interval 
    x_axis, y_axis = np.linspace(0, 20, 10), np.linspace(0, 20, 10) 

    # Create a rectangular grid out of 
    # two given one-dimensional arrays 
    X, Y = np.meshgrid(x_axis, y_axis) 

    # Make a rectangular grid 
    # 3-dimensional by calculating z1, z2, z3 
    Z1 = (w1-x1*X-y1*Y)/z1 
    Z2 = (w2-x2*X-y2*Y)/z2 
    Z3 = (w3+X-Y)/z3 

    # Create 3D graphics and add 
    # an add an axes to the figure 
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') 

    # Create a 3D Surface Plot 
    ax.plot_surface(X, Y, Z1, alpha=0.7, 
                    cmap=cm.Accent, 
                    rstride=100, cstride=100) 
    ax.plot_surface(X, Y, Z2, alpha=0.7, 
                    cmap=cm.Paired, 
                    rstride=100, cstride=100) 
    ax.plot_surface(X, Y, Z3, alpha=0.7, 
                    cmap=cm.Pastel1, 
                    rstride=100, cstride=100,) 

    # Draw points and make lines 
    ax.plot((sol[0],), (sol[1],), (sol[2],), 
            lw=2, c='k', marker='o', 
            markersize=7, markeredgecolor='g', 
            markerfacecolor='white') 

    # Set the label for x-axis, y-axis and 
    # z-axis 
    ax.set_xlabel('X axis') 
    ax.set_ylabel('Y axis') 
    ax.set_zlabel('Z axis') 

    # Display all figures 
    #plt.show() 
    plt.savefig("my_politope.png")
