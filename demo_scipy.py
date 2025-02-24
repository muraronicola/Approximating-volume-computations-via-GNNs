from scipy.spatial import HalfspaceIntersection
import numpy as np
halfspaces = np.array([    [1, 0, 0, -1],  # x <= 1
    [-1, 0, 0, -1], # x >= -1
    [0, 1, 0, -1],  # y <= 1
    [0, -1, 0, -1], # y >= -1
    [0, 0, 1, -1],  # z <= 1
    [0, 0, -1, -1]  # z >= -1
    ])
feasible_point = np.array([0, 0, 0])
hs = HalfspaceIntersection(halfspaces, feasible_point)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xlim, ylim, zlim = (-1, 3), (-1, 3), (-1, 3)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlim(xlim)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

x = np.linspace(-1, 3, 100)
y = np.linspace(-1, 3, 100)
symbols = ['-', '+', 'x', '*']
signs = [0, 0, -1, -1]
fmt = {"color": None, "edgecolor": "b", "alpha": 0.5}
for h, sym, sign in zip(halfspaces, symbols, signs):
    hlist = h.tolist()
    fmt["hatch"] = sym
    if h[1]== 0:
        ax.axvline(-h[2]/h[0], label='{}x+{}y+{}=0'.format(*hlist))
        #ax.plot_surface(X, Y + Y_offset, Z + Z_offset, rstride=1, cstride=1, facecolors=facecolors, shade=False)
        xi = np.linspace(xlim[sign], -h[2]/h[0], 100)
        yi = np.linspace(ylim[sign], -h[3]/h[0], 100)
        #print(xi, yi)
        print("Ciao1")
        print("xi", xi)
        ax.fill_between(xi, ylim[0], ylim[1], yi, zlim[0], zlim[1], **fmt)
    else:
        ax.plot(x, (-h[2]-h[0]*x)/h[1], label='{}x+{}y+{}=0'.format(*hlist))
        
        #now in 3d
        #ax.plot_surface(x, y, (-h[2]-h[0]*x-h[1]*y)/h[2], alpha=0.5)
        ax.fill_between(x, (-h[2]-h[0]*x)/h[1], ylim[sign], y, (-h[2]-h[0]*y)/h[1], zlim[sign], **fmt)
        
        print("Ciao2")
        
        #ax.fill_between(x, (-h[2]-h[0]*x)/h[1], ylim[sign], **fmt)
x, y, z = zip(*hs.intersections)
ax.plot(x, y, z, 'o', markersize=8, color='r', label="Border Point")
plt.show()


"""
from scipy.optimize import linprog
from matplotlib.patches import Circle
norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1),
    (halfspaces.shape[0], 1))
c = np.zeros((halfspaces.shape[1],))
c[-1] = -1
A = np.hstack((halfspaces[:, :-1], norm_vector))
b = - halfspaces[:, -1:]
res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
x = res.x[:-1]
y = res.x[-1]
#circle = Circle(x, radius=y, alpha=0.3)
#ax.add_patch(circle)
plt.legend(bbox_to_anchor=(1.6, 1.0))
plt.show()
"""