import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import HalfspaceIntersection, ConvexHull, QhullError

def generate_random_h_representation(dim=3, num_ineq=10, seed=None):
    """
    Genera casualmente una rappresentazione H di un politopo in uno spazio di dimensione dim.
    """
    if seed is not None:
        np.random.seed(2)

    # Generiamo piani casuali con normali casuali e shift casuale
    A = np.random.randn(num_ineq, dim)
    b = np.random.rand(num_ineq) * 5  # Per evitare politopi degeneri, shift positivo

    return A, b

def compute_vertices(A, b):
    """
    Trova i vertici del politopo dato un insieme di disuguaglianze Ax <= b.
    """
    num_ineq, dim = A.shape
    vertices = []
    
    # Troviamo tutti i sottoinsiemi di equazioni che potrebbero definire un vertice
    from itertools import combinations
    for idxs in combinations(range(num_ineq), dim):  # dim equazioni per trovare un punto
        try:
            A_sub = A[list(idxs)]
            b_sub = b[list(idxs)]
            vertex = np.linalg.solve(A_sub, b_sub)  # Risolve Ax = b
            if np.all(A @ vertex <= b + 1e-6):  # Controlla se il punto soddisfa tutte le altre disuguaglianze
                vertices.append(tuple(vertex))
        except np.linalg.LinAlgError:
            continue  # Skip if the system is singular
    
    return list(set(vertices))  # Rimuoviamo duplicati

def build_graph(vertices):
    """
    Costruisce un grafo in cui i nodi sono i vertici e gli spigoli sono definiti dalla loro vicinanza.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(vertices)))

    # Connettiamo i vertici se la loro distanza è "piccola" (euristica)
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            if i < j and np.linalg.norm(np.array(v1) - np.array(v2)) < 2.0:  # Soglia arbitraria
                G.add_edge(i, j)

    return G

def plot_graph_3d(vertices, G):
    """
    Plotta il grafo in 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    pos = {i: v for i, v in enumerate(vertices)}
    for i, j in G.edges:
        v1, v2 = np.array(vertices[i]), np.array(vertices[j])
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 'k-')

    for i, v in enumerate(vertices):
        ax.scatter(v[0], v[1], v[2], c='r', s=50)
    
    #plt.show()
    plt.savefig("graph.png")

# Generiamo un politopo casuale in 3D
A, b = generate_random_h_representation(dim=3, num_ineq=10, seed=42)
vertices = compute_vertices(A, b)

if len(vertices) > 0:
    G = build_graph(vertices)
    plot_graph_3d(vertices, G)
else:
    print("Il sistema di vincoli non genera un politopo valido.")
