



# Import some useful modules.
import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt
import csv
import time
import gc
import psutil
import pandas as pd
from jax.interpreters import xla

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh


# Define constitutive relationship.
class HyperElasticity_opt(Problem):

    def custom_init(self):
        self.fe = self.fes[0]
        tol = 1e-8
        self.fixed_bc_mask = np.abs(mesh.points[:, 0] - 0) < tol  # Use NumPy here
        self.non_fixed_indices = np.where(~self.fixed_bc_mask)[0]
        self.np_points = np.array(self.mesh[0].points)
    def get_tensor_map(self):

        def psi(F_tilde):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            # J = np.linalg.det(F)
            # Jinv = J**(-2. / 3.)
            # I1 = np.trace(F.T @ F)

            J = np.linalg.det(F_tilde)
            Jinv = J**(-2. / 3.)
            I1 = np.trace(F_tilde.T @ F_tilde)

            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad,u_grads_0):
            I = np.eye(self.dim)
            F = u_grad + I
            F_0 = u_grads_0 + I
            F_0_inv = np.linalg.inv(F_0)
            F_tilde = np.dot(F, F_0_inv)
            P = P_fn(F_tilde)
            # jax.debug.print("u_grads_0: {}", u_grads_0)

            return P
        
        return first_PK_stress


    # Define the source term b
    def get_mass_map(self):
        def mass_map(u, x):
            val = np.array([0.0, 0.0, 0.1]) 
            return val
        return mass_map

    def set_params(self, params):
        # self.X_0 = self.mesh[0].points + params

        reconstructed_param = np.zeros_like(self.np_points)
        reconstructed_param = reconstructed_param.at[self.non_fixed_indices].set(params)
        # reconstructed_param = reconstructed_param.at[~fixed_bc_mask].set(params)

        self.X_0 = self.np_points + reconstructed_param
        # jax.debug.print("X_0: {}", self.X_0)
        # self.params
        self.params = reconstructed_param

        self.__post_init__()






class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.

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

    # Define the source term b
    def get_mass_map(self):
        def mass_map(u, x):
            val = np.array([0.0, 0.0, 0.1]) 
            return val
        return mass_map



# Specify mesh-related information (first-order hexahedron element).

ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 3., 3., 3.
meshio_mesh = box_mesh_gmsh(Nx=3,
                            Ny=3,
                            Nz=3,
                            Lx=Lx,
                            Ly=Ly,
                            Lz=Lz,
                            data_dir=data_dir,
                            ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# print(mesh.points)


# Define boundary locations.
def left(point):
    # print(np.isclose(point[0], Lx, atol=1e-5))
    return np.isclose(point[0], 0, atol=1e-5)

# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[left] *3, # Results in [left, left, left] + [right, right, right] 
                     [0, 1, 2], # Components: [u_x, u_y, u_z] 
                     [zero_dirichlet_val] * 3]


problem_2 = HyperElasticity(mesh,
                            vec=3,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info
)

sol_list_2 = solver(problem_2, solver_options={'petsc_solver': {}})

# Store the solution
u_sol_2 = sol_list_2[0]
print(u_sol_2)
# vtk_path_2 = os.path.join(data_dir, 'vtk', 'u_one_node.vtu')
# os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
# save_sol(problem_2.fes[0], u_sol_2, vtk_path_2)

# Create an instance of the problem.





# mesh_points_modi = mesh.points 8.79406 -3.2669
# scale_d = 1.
# original_cood = np.copy(mesh.points) * 1.1


observed_positions_2 = mesh.points + u_sol_2
undeformed_coord = mesh.points
original_cood = observed_positions_2

mesh.points = onp.array(observed_positions_2)


problem = HyperElasticity_opt(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
# params = np.zeros_like(problem.mesh[0].points)
# params = np.ones_like(problem.mesh[0].points)



params = np.array(original_cood) * 0 # 0.0001
tol = 1e-8
fixed_bc_mask = np.abs(mesh.points[:,0] - 0) < tol
non_fixed_indices = np.where(~fixed_bc_mask)[0]
params = params[non_fixed_indices]


print(params)
print("HAHA")
# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem)
print("HOHO")
sol_list = fwd_pred(params)
# print("sol_list")
# print(sol_list[0])

def test_fn(sol_list):
    print('test fun')
    # print(sol_list[0])
    # jax.debug.print("cost func: {}", np.sum((sol_list[0] - u_sol_2)**2))
    return np.sum((((sol_list[0]+problem.mesh[0].points) - observed_positions_2))**2)  #/np.sum((observed_positions_2)**2)) #np.sum((sol_list[0] - u_sol_2)**2)
    #Set parameter without fixed nodes.
    #Normalize
     
def composed_fn(params):
    return np.sum(test_fn(fwd_pred(params))) #test_fn(fwd_pred(params))


d_coord= jax.grad(composed_fn)(params)


print("d_coord")
print(d_coord.shape)
print(d_coord)



def finite_diff_grad(fn, params, epsilon=1e-6):
    grads = np.zeros_like(params)
    for i in range(3):
        for j in range(params.shape[1]):
            params_pos = params.at[i,j].add(epsilon)
            params_neg = params.at[i,j].add(-epsilon)
            grads = grads.at[i,j].set((fn(params_pos) - fn(params_neg)) / (2 * epsilon))
    return grads

finite_grad = finite_diff_grad(lambda p: np.sum((fwd_pred(p)[0] - u_sol_2)**2), params)
print("Finite difference gradient:", finite_grad)
print("d_coord")
print(d_coord.shape)
print(d_coord)