import jax
import jax.numpy as np
import os
from jax import Array
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
import scipy.optimize
from jax_fem.solver import ad_wrapper
from jaxopt import ScipyMinimize
from jax import Array
import numpy as onp
from jax import device_get

# Define constitutive relationship.
class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.

    def __init__(self, *args, internal_pressure=2.0, **kwargs):
        self.internal_pressure = internal_pressure  # Make internal pressure variable

        # # Extract mesh from args
        # mesh = args[0]

        # # Check if mesh.points is a JAX array and convert to NumPy array if necessary
        # if isinstance(mesh.points, Array):
        #     mesh.points = onp.array(mesh.points)

        # # Reconstruct args with the potentially modified mesh
        # args = (mesh,) + args[1:]

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
            normal = np.array([-1.0, 0.0, 0.0])
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
meshio_mesh = box_mesh_gmsh(Nx=3,
                            Ny=3,
                            Nz=3,
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

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)



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
vtk_path_2 = os.path.join(data_dir, 'vtk', 'u_one_node.vtu')
os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
save_sol(problem_2.fes[0], u_sol_2, vtk_path_2)


# print(u_sol_2)

# -------------------------------------------------------------------------------------------------------------
# Forward for pressure = 3-------------------------------------------------------------------------------------

internal_pressure_3 = 3.0

problem_3 = HyperElasticity(
    mesh,
    vec=3,
    dim=3,
    ele_type=ele_type,
    dirichlet_bc_info=dirichlet_bc_info,
    location_fns=location_fns,
    internal_pressure=internal_pressure_3
)

# Solve the problem
sol_list_3 = solver(problem_3, solver_options={'petsc_solver': {}})

# Store the solution
u_sol_3 = sol_list_3[0]
vtk_path_3 = os.path.join(data_dir, 'vtk', 'u_pressure_3.vtu')
os.makedirs(os.path.dirname(vtk_path_3), exist_ok=True)
save_sol(problem_3.fes[0], u_sol_3, vtk_path_3)

# print(u_sol_3)


# -------------------------------------------------------------------------------------------------------------
# --------------------------------------------Inverse Problem Starts ------------------------------------------
# -------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------
# Data-set ----------------------------------------------------------------------------------------------------
observed_positions_2 = meshio_mesh.points + onp.array(u_sol_2)
observed_positions_3 = meshio_mesh.points + onp.array(u_sol_3)
original_positions   = np.array(meshio_mesh.points)


# Identify fixed nodes (nodes on 'right' boundary)
fixed_nodes = onp.isclose(meshio_mesh.points[:, 0], Lx, atol=1e-5)
non_fixed_nodes = ~fixed_nodes
# -------------------------------------------------------------------------------------------------------------
# Lose Function -----------------------------------------------------------------------------------------------
# @jit
ii = 1

def loss_function(init_non_fixed_nodes_flat):
    global ii
    print(ii)
    # Reshape the flat array back to node positions
    # original_nodes = original_nodes_flat.reshape(meshio_mesh.points.shape)

    original_nodes = original_positions.at[non_fixed_nodes].set(init_non_fixed_nodes_flat.reshape((-1, 3)))


    # Create a mesh with the current estimate of the original positions
    current_mesh = Mesh(original_nodes, mesh.cells)
    
    # Set up and solve the forward problem for pressure = 2
    problem_2 = HyperElasticity(
        current_mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        internal_pressure=internal_pressure_2
    )
    # sol_list_2 = jax.lax.stop_gradient(solver(problem_2, solver_options={'petsc_solver': {}}))
    sol_list_2 = solver(problem_2, solver_options={'petsc_solver': {}})
    u_sol_2 = sol_list_2[0]
    simulated_positions_2 = original_nodes + u_sol_2
    
    # Set up and solve the forward problem for pressure = 3
    problem_3 = HyperElasticity(
        current_mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        internal_pressure=internal_pressure_3
    )
    # sol_list_3 = jax.lax.stop_gradient(solver(problem_3, solver_options={'petsc_solver': {}}))
    sol_list_3 = solver(problem_3, solver_options={'petsc_solver': {}})
    u_sol_3 = sol_list_3[0]
    simulated_positions_3 = original_nodes + u_sol_3
    
    # Compute the loss as the sum of squared differences
    # @jit
    # loss_2 = np.sum((simulated_positions_2 - observed_positions_2)**2)
    # loss_3 = np.sum((simulated_positions_3 - observed_positions_3)**2)
    # total_loss = loss_2 + loss_3

    loss_2 = np.sum((simulated_positions_2[non_fixed_nodes] - observed_positions_2[non_fixed_nodes])**2)
    loss_3 = np.sum((simulated_positions_3[non_fixed_nodes] - observed_positions_3[non_fixed_nodes])**2)
    regularization = onp.sum((original_nodes - mesh_points_onp)**2)
    total_loss = regularization

    print("total_loss: ")
    print(total_loss)

    return total_loss


# loss_and_grad = value_and_grad(loss_function)


# -------------------------------------------------------------------------------------------------------------
# Initial Guess -----------------------------------------------------------------------------------------------
print('jkjk')
init_non_fixed_nodes_flat = observed_positions_2[non_fixed_nodes].flatten()
# initial_nodes_flat = observed_positions_3.flatten()
# initial_nodes_flat = original_positions.flatten()
# if isinstance(init_non_fixed_nodes_flat, Array):
#     print("This is a JAX array.")

# # Check if it is a NumPy array
# if isinstance(init_non_fixed_nodes_flat, onp.ndarray):
#     print("This is a NumPy array.")
# Define the objective function for optimization
# @jit
def objective_function(x):
    return loss_function(x)

optimizer = ScipyMinimize(fun=objective_function)

opt_results = optimizer.run(init_params=init_non_fixed_nodes_flat)

optimized_original_nodes_flat = opt_results.params

optimized_original_nodes = optimized_original_nodes_flat.reshape(meshio_mesh.points.shape)

# Display the results
print("Optimization complete.")
print("Optimized original node positions:")
print(optimized_original_nodes)


# def optimize_original_nodes():
#     result = scipy.optimize.minimize(
#         fun=loss_function,
#         x0=initial_nodes_flat,
#         # jac=True,
#         method= 'Nelder-Mead', #'Powell',
#         # method='L-BFGS-B',      # Gradient-based optimizer
#         # jac='2-point', 
#         options={'disp': True, 'maxiter': 100}
#     )
#     return result.x.reshape(meshio_mesh.points.shape)

# # Run the optimization
# estimated_original_nodes = optimize_original_nodes()


# print(estimated_original_nodes)
