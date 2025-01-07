# Import some useful modules.
import jax
import jax.numpy as np
import os

from jax import jit
# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from jax import value_and_grad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax.scipy.optimize import minimize
from jaxopt import GradientDescent
from jax import Array
import numpy as onp
# Define constitutive relationship.
class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.

    def __init__(self, *args, internal_pressure=2.0, **kwargs):
        self.internal_pressure = internal_pressure  # Make internal pressure variable
        super().__init__(*args, **kwargs)

    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
        
        return first_PK_stress

    def get_surface_maps(self):
        internal_pressure = self.internal_pressure  # N/m²   (t = -pn)

        def x_0_traction(u, x):
            normal = np.array([-5.0, 0.0, 0.0])
            return -internal_pressure * normal
        def y_0_traction(u, x):
            normal = np.array([0., -1.0, 0.0])
            return -internal_pressure * normal
        def y_1_traction(u, x):
            normal = np.array([0.0, 1.0, 0.0])
            return -internal_pressure * normal
        def z_0_traction(u, x):
            normal = np.array([0.0, 0.0, -1.0])
            return -internal_pressure * normal
        def z_1_traction(u, x):
            normal = np.array([0.0, 0.0, 1.0])
            return -internal_pressure * normal
        
        # def x_left_traction(u, x):
        #     normal = np.array([-1.0, 0, 0])
        #     return -internal_pressure * normal
        
        return [x_0_traction, y_0_traction, y_1_traction, z_0_traction, z_1_traction]


# 

# Specify mesh-related information (first-order hexahedron element).
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh_gmsh(Nx=30,
                            Ny=30,
                            Nz=30,
                            Lx=Lx,
                            Ly=Ly,
                            Lz=Lz,
                            data_dir=data_dir,
                            ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])



corner_point1 = np.array([0.0, 0.0, 1.0])
corner_point2 = np.array([0.0, 1.0, 1.0])

HHY1 = np.where(np.all(mesh.points==corner_point1,axis=1))[0][0]
HHY2 = np.where(np.all(mesh.points==corner_point2,axis=1))[0][0]

# HHY1 = np.where(mesh.points[:,0]==1.0)[0]
# HHY = np.array([HHY1,HHY2])
# HHY = np.array([HHY1,HHY2])
traction_x_0_point = np.where(mesh.points[:,0]==0)[0]
traction_y_0_point = np.where(mesh.points[:,1]==0)[0]
traction_y_1_point = np.where(mesh.points[:,1]==Ly)[0]
traction_z_0_point = np.where(mesh.points[:,2]==0)[0]
traction_z_1_point = np.where(mesh.points[:,2]==Lz)[0]
# print(HHY)

# Define boundary locations for traction.
def traction_x_0(point, ind):
    # print(ind, ind_set_traction)
    return np.isin(ind, traction_x_0_point) 

def traction_y_0(point, ind):
    # print(ind, ind_set_traction)
    return np.isin(ind, traction_y_0_point) 
def traction_y_1(point, ind):
    # print(ind, ind_set_traction)
    return np.isin(ind, traction_y_1_point) 
def traction_z_0(point, ind):
    # print(ind, ind_set_traction)
    return np.isin(ind, traction_z_0_point) 
def traction_z_1(point, ind):
    # print(ind, ind_set_traction)
    return np.isin(ind, traction_z_1_point) 


# Define boundary locations.
# def left(point):
    # return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

# def y_down(point):
#     return np.isclose(point[1], 0, atol=1e-5)

# def y_up(point):
#     return np.isclose(point[1], Ly, atol=1e-5)

# def z_down(point):
#     return np.isclose(point[2], 0, atol=1e-5)

# def z_up(point):
#     return np.isclose(point[2], Lz, atol=1e-5)




# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.


# def dirichlet_val_x2(point):
#     return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
#             (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.


# def dirichlet_val_x3(point):
#     return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
#             (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.


# dirichlet_bc_info = [[left] * 3 + [right] * 3, # Results in [left, left, left] + [right, right, right] 
#                      [0, 1, 2] + [0, 1, 2], # Components: [u_x, u_y, u_z] 
#                      [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] + [zero_dirichlet_val] * 3]

dirichlet_bc_info = [[right] + [right] + [right], # Results in [left, left, left] + [right, right, right] 
                     [0, 1, 2], # Components: [u_x, u_y, u_z] 
                     [zero_dirichlet_val] + [zero_dirichlet_val] + [zero_dirichlet_val]]

location_fns = [traction_x_0, traction_y_0, traction_y_1, traction_z_0, traction_z_1]

# -------------------------------------------------------------------------------------------------------------
# --------------------------------------------Generating Data-set ---------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------
# Forward for pressure = 2-------------------------------------------------------------------------------------
internal_pressure_2 = 2.0

problem_2 = HyperElasticity(mesh,
                            vec=3,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns=location_fns,
                            internal_pressure=internal_pressure_2
)

# Solve the problem
sol_list_2 = solver(problem_2, solver_options={'petsc_solver': {}})

# Store the solution
u_sol_2 = sol_list_2[0]
vtk_path_2 = os.path.join(data_dir, 'vtk', 'u_one_node_2.vtu')
os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
save_sol(problem_2.fes[0], u_sol_2, vtk_path_2)


print(f"\n\nlocation_fns1 is processed to generate Dirichlet node indices: \n{problem_2.fes[0].node_inds_list}")
print(f"\nwhere node_inds_list[l][j] returns the jth selected node index in Dirichlet set l")

print(f"\n\nlocation_fns is processed to generate boundary indices list: \n{problem_2.boundary_inds_list}")
print(f"\nwhere boundary_inds_list[k][i, 0] returns the global cell index of the ith selected face of boundary subset k")
print(f"      boundary_inds_list[k][i, 1] returns the local face index of the ith selected face of boundary subset k")



# fixed_nodes = np.isclose(meshio_mesh.points[:, 0], Lx, atol=1e-5)
# non_fixed_nodes = ~fixed_nodes
# print(fixed_nodes)

# fixed_nodes_1 = meshio_mesh.points[fixed_nodes]
# non_fixed_nodes_1 = meshio_mesh.points[non_fixed_nodes]

# print(non_fixed_nodes_1)
# print(non_fixed_nodes_1.shape)
# observed_positions_2 = meshio_mesh.points + onp.array(u_sol_2)
# Check if it is a JAX array
if isinstance(observed_positions_2, Array):
    print("This is a JAX array.")

# Check if it is a NumPy array
if isinstance(observed_positions_2, onp.ndarray):
    print("This is a NumPy array.")

###############################################################################
############################  Visualization  ##################################
###############################################################################
# points = meshio_mesh.points
# cells = meshio_mesh.cells_dict[cell_type]  # Shape: (num_elements, nodes_per_element)

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b', s=50)

# for i, (x, y, z) in enumerate(points):
#     ax.text(x, y, z, f'{i}', color='red', fontsize=10)

# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# ax.set_zlabel('Z Coordinate')
# ax.set_title('Mesh Nodes with Node Numbers')
# plt.show()

###############################################################################
###############################################################################
###############################################################################