import jax
import jax.numpy as np
import os
import pandas as pd
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
import numpy as onp
from jax import grad
from jax import Array
from jax_fem.mma import optimize

# Define constitutive relationship.
class HyperElasticity(Problem):
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
        return lambda u_grad: P_fn(u_grad + np.eye(self.dim))

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

# Mesh setup
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh_gmsh(Nx=3, Ny=3, Nz=3, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Boundary conditions
# Define boundary locations.

def right(point):
    # print(np.isclose(point[0], Lx, atol=1e-5))
    return np.isclose(point[0], Lx, atol=1e-5)

# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[right] + [right] + [right], # Results in [left, left, left] + [right, right, right] 
                     [0, 1, 2], # Components: [u_x, u_y, u_z] 
                     [zero_dirichlet_val] + [zero_dirichlet_val] + [zero_dirichlet_val]]

traction_x_0_point = np.where(mesh.points[:,0]==0)[0]
traction_y_0_point = np.where(mesh.points[:,1]==0)[0]
traction_y_1_point = np.where(mesh.points[:,1]==Ly)[0]
traction_z_0_point = np.where(mesh.points[:,2]==0)[0]
traction_z_1_point = np.where(mesh.points[:,2]==Lz)[0]
# print(HHY)

# Define boundary locations for traction.
def traction_x_0(point, ind):
    # print(np.isin(ind, traction_x_0_point) )
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


location_fns = [traction_x_0, traction_y_0, traction_y_1, traction_z_0, traction_z_1]

# Observed positions from the true initial conditions
problem_2 = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,location_fns=location_fns, internal_pressure=2.0)
observed_positions_2 = mesh.points + solver(problem_2)[0]
problem_3 = HyperElasticity(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,location_fns=location_fns, internal_pressure=3.0)
observed_positions_3 = mesh.points + solver(problem_3)[0]

fixed_nodes = np.isclose(mesh.points[:, 0], Lx, atol=1e-5)
non_fixed_nodes = ~fixed_nodes

mesh_points = mesh.points
mesh_cells = mesh.cells

# Define Forward Problems
def forward_problem_2(initial_displacements, mesh_cells, ele_type, dirichlet_bc_info, location_fns):
    current_mesh = Mesh(initial_displacements, mesh_cells)
    problem_current_2 = HyperElasticity(current_mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,location_fns=location_fns, internal_pressure=2.0)
    fwd_pred_current_2 = ad_wrapper(problem_current_2)
    sol_list_2 = fwd_pred_current_2()
    return sol_list_2[0]

def forward_problem_3(initial_displacements, mesh_cells, ele_type, dirichlet_bc_info, location_fns):
    current_mesh = Mesh(initial_displacements, mesh_cells)
    problem_current_3 = HyperElasticity(current_mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,location_fns=location_fns, internal_pressure=3.0)
    fwd_pred_current_3 = ad_wrapper(problem_current_3)
    sol_list_3 = fwd_pred_current_3()
    return sol_list_3[0]

# Define Loss Function
def loss_function(non_fixed_nodes_flat, mesh_points, mesh_cells, ele_type, dirichlet_bc_info, location_fns, observed_positions_2, observed_positions_3):
    initial_displacements = mesh_points.copy()
    initial_displacements[non_fixed_nodes] = non_fixed_nodes_flat.reshape((-1, 3))

    u_sol_2 = forward_problem_2(initial_displacements, mesh_cells, ele_type, dirichlet_bc_info, location_fns)
    u_sol_3 = forward_problem_3(initial_displacements, mesh_cells, ele_type, dirichlet_bc_info, location_fns)

    simulated_positions_2 = initial_displacements + u_sol_2
    simulated_positions_3 = initial_displacements + u_sol_3

    # Compute the loss using only non-fixed nodes
    loss_2 = np.sum((simulated_positions_2[non_fixed_nodes] - observed_positions_2[non_fixed_nodes])**2)
    loss_3 = np.sum((simulated_positions_3[non_fixed_nodes] - observed_positions_3[non_fixed_nodes])**2)
    regularization = np.sum((initial_displacements - mesh_points)**2)

    total_loss = loss_2 + loss_3 + 0.1 * regularization
    print("Total Loss:", total_loss)
    return total_loss

# # Objective Function for MMA
# def objectiveHandle(non_fixed_nodes_flat):
#     total_loss, total_grad = jax.value_and_grad(loss_function)(non_fixed_nodes_flat, mesh_points, mesh_cells, ele_type, dirichlet_bc_info, location_fns, observed_positions_2, observed_positions_3)
#     return total_loss, total_grad


def objectiveHandle(non_fixed_nodes_flat):
    J, dJ = jax.value_and_grad(J_total)(non_fixed_nodes_flat)
    return J, dJ

























# ############################

# def objectiveHandle(non_fixed_nodes_flat):
#     # Call `loss_function` and compute its gradient
#     total_loss, total_grad = jax.value_and_grad(loss_function)(
#         non_fixed_nodes_flat,
#         mesh_points,
#         mesh_cells,
#         ele_type,
#         dirichlet_bc_info,
#         location_fns,
#         observed_positions_2,
#         observed_positions_3
#     )
#     return total_loss, total_grad

# # # Constraint Function for MMA (if needed)
# # def consHandle(non_fixed_nodes_flat):
# #     # Implement your constraints here
# #     c = ...  # Constraint value
# #     gradc = ...  # Gradient of the constraint
# #     return c, gradc

# def consHandle(non_fixed_nodes_flat):
#     # Reshape the input flat array back to node positions for calculation
#     current_displacements = non_fixed_nodes_flat.reshape((-1, 3))
    
#     # Example constraint: Limit the average displacement to a threshold
#     avg_displacement = np.mean(np.linalg.norm(current_displacements, axis=1))
#     max_allowed_displacement = 0.1  # Set your constraint threshold

#     # Constraint value g = avg_displacement / max_allowed_displacement - 1
#     # If g <= 0, the constraint is satisfied
#     g = avg_displacement / max_allowed_displacement - 1.0

#     # Compute the gradient of the constraint with respect to node positions
#     g_grad = jax.grad(lambda x: np.mean(np.linalg.norm(x.reshape((-1, 3)), axis=1)))(
#         non_fixed_nodes_flat
#     )

#     # Return the constraint as required by MMA, as a tuple: (constraint value, gradient)
#     return np.array([g]), np.array([g_grad])



# # Optimization Parameters
# vf = 0.5
# optimizationParams = {'maxIters': 100, 'movelimit': 0.1}
# rho_ini = observed_positions_3[non_fixed_nodes].flatten()
# numConstraints = 1

# optimize(
#     problem=None,  # If your MMA implementation requires it, set appropriately
#     x0=rho_ini,
#     optimizationParams=optimizationParams,
#     objectiveHandle=objectiveHandle,  # Here we pass `objectiveHandle` that uses `loss_function`
#     consHandle=consHandle,  # If you have constraints, otherwise set as needed
#     numConstraints=numConstraints
# )

# # optimize(
# #     problem=None,
# #     x0=initial_nodes_flat,
# #     optimizationParams=optimizationParams,
# #     objectiveHandle=objectiveHandle,  # Using loss_function indirectly
# #     consHandle=consHandle,  # Defined constraint function
# #     numConstraints=1  # Set to 1 for this single constraint
# # )