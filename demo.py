import cdd

# Define the H-representation (Ax ≤ b) for a cube (unit cube [0,1]^3)
h_rep = [
    [1, 1, 0, 0],  # x ≥ 0
    [1, 0, 1, 0],  # y ≥ 0
    [1, 0, 0, 1],  # z ≥ 0
    [1, -1, 0, 0], # x ≤ 1
    [1, 0, -1, 0], # y ≤ 1
    [1, 0, 0, -1]  # z ≤ 1
]

# Convert to cdd Matrix
mat = cdd.Matrix(h_rep)
mat.rep_type = cdd.RepType.INEQUALITY  # Define it as H-representation

# Create polytope
poly = cdd.Polyhedron(mat)

# Get vertices and facets
vertices = poly.get_generators()  # Extreme points (vertices)
facets = poly.get_inequalities()  # Inequalities defining facets

# Count elements
num_vertices = sum(1 for row in vertices if row[0] == 1)  # Count vertices
num_facets = len(facets)  # Facets (2D faces)

# Approximate edges using combinatorial formula for convex polytopes: E = V + F - D - 1
# For a 3D polytope, E = V + F - 4
num_edges = num_vertices + num_facets - 4

# Print results
print(f"Number of 0-dimensional faces (vertices): {num_vertices}")
print(f"Number of 1-dimensional faces (edges): {num_edges}")
print(f"Number of 2-dimensional faces (facets): {num_facets}")
print(f"Number of 3-dimensional faces (the whole polytope): 1")
