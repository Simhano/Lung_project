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
import meshio
import bisect

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh
import jax.nn


# Define constitutive relationship.
class HyperElasticity_opt(Problem):
    def __init__(self, *args,density=2.0, fixed_bc_mask = None , **kwargs):
        self.density = density  # Make internal pressure variable
        self.fixed_bc_mask = fixed_bc_mask
        super().__init__(*args, **kwargs)

    def custom_init(self):
        self.fe = self.fes[0]
        tol = 1e-8
        # self.fixed_bc_mask = np.abs(mesh.points[:, 1] - 0) < tol  # Use NumPy here
        # self.fixed_bc_mask = mesh.points[:, 1] >= -18/1000
        self.non_fixed_indices = np.where(~self.fixed_bc_mask)[0]
        self.np_points = np.array(self.mesh[0].points)
        self.param_flag = 0
    def get_tensor_map(self):

        def psi(F_tilde):
            E = 2.842800754666559e+03
            nu = 0.48
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            # J = np.linalg.det(F)
            # Jinv = J**(-2. / 3.)
            # I1 = np.trace(F.T @ F)

            J = np.linalg.det(F_tilde)
            # J_min = np.min(J)
            # if J_min < 0:
            #     jax.debug.print("J_min: {}", J_min)

            Jinv = J**(-2. / 3.)
            I1 = np.trace(F_tilde.T @ F_tilde)

            energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.


            # Add penalty if J falls below a threshold (to discourage inversion)
            threshold = 0.5
            penalty_weight = 1e3

            # Using a smooth approximation: softplus gives a smooth penalty
            penalty = penalty_weight * jax.nn.softplus(threshold - J)
            return energy
            # return energy + penalty

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad,u_grads_0):
            I = np.eye(self.dim)
            F = u_grad + I
            F_0 = u_grads_0 + I
            F_0_inv = np.linalg.inv(F_0)
            F_tilde = np.dot(F, F_0_inv)
            J_0 = np.linalg.det(F_0)
            P = J_0 * np.matmul(P_fn(F_tilde) , np.transpose(F_0_inv))
            # jax.debug.print("u_grads_0: {}", u_grads_0)
            return P
        
        return first_PK_stress


    # Define the source term b
    def get_mass_map(self):
        def mass_map(u, x,F_0_S):
            density = self.density  # kg/m³, breast tissue density
            g = 9.81  # m/s², gravitational acceleration
            # val = np.array([0.0, -density*g, 0.0])
            # val = np.array([0.0, 0.0, density*g])
            val = np.array([0.0, density*g])
            # jax.debug.print("density: {}", density)
            return val
        return mass_map

    def set_params(self, params):

        self.np_points = np.array(self.mesh[0].points)
        reconstructed_param = np.zeros_like(self.np_points)
        reconstructed_param = reconstructed_param.at[self.non_fixed_indices].set(params)

        self.X_0 = self.np_points + reconstructed_param
        # jax.debug.print("X_0: {}", self.X_0)

        self.params = reconstructed_param

        self.__post_init__()

    def penalty_fn(self, sol_list, params):

        # self.X_0 = self.mesh[0].points + params
        np_points = np.array(self.mesh[0].points)
        reconstructed_param = np.zeros_like(np_points)
        reconstructed_param = reconstructed_param.at[self.non_fixed_indices].set(params)

        X_0 = np_points + reconstructed_param

        sol_list_0 = [X_0 - self.mesh[0].points]
        # cells_sol_list_0 = [sol[cells] for cells, sol in zip(self.cells_list, sol_list_0)] # [(num_cells, num_nodes, vec), ...]
        # cells_sol_flat_0 = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list_0) # (num_cells, num_nodes*vec + ...)

        # cells_sol_list = [sol[cells] for cells, sol in zip(self.cells_list, sol_list)] # [(num_cells, num_nodes, vec), ...]
        # cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*cells_sol_list) # (num_cells, num_nodes*vec + ...)


        u_grads = np.take(sol_list[0], self.fe.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim) 

        u_grads_0 = np.take(sol_list_0[0], self.fe.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads_0 = np.sum(u_grads_0, axis=2) # (num_cells, num_quads, vec, dim) 


        I = np.eye(2)                      # (dim, dim)
        I = I.reshape((1, 1, 2, 2))      # (1, 1, dim, dim)

        F = u_grads + I                      # (n_cells, n_quads, dim, dim)

        F_0 = u_grads_0 + I                      # (n_cells, n_quads, dim, dim)
        # jax.debug.print("F: {}", F)


        F_0_inv = np.linalg.inv(F_0)  # still (c, q, i, j)

        #    i,j shape labels:   c=cell, q=quad, i,j,k = dim indices
        F_tilde = np.einsum('c q i j, c q j k -> c q i k',
                            F, F_0_inv)   # -> (c, q, i, k)
        
        #    'cqji, cqjk -> cqi k'
        C_tilde = np.einsum('cqji, cqjk -> cqik',
                            F_tilde, F_tilde)      # (n_cells, n_quads, dim, dim)

        # trace over the last two dims → (n_cells, n_quads)
        trace_C = np.trace(C_tilde, axis1=2, axis2=3)

        # volumetric strain
        eps_v = 0.5 * (trace_C - 2)        # shape (n_cells, n_quads)

        # square and then average over *all* cells & quads
        strain_sq_mean = np.mean(eps_v**2)

        jax.debug.print("strain_sq_mean: {}", strain_sq_mean)

        return strain_sq_mean




class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.

    def get_tensor_map(self):

        def psi(F):
            E = 2.842800754666559e+03
            nu = 0.48
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
            density = 942.8238474  # kg/m³, breast tissue density
            g = 9.81  # m/s², gravitational acceleration
            # val = np.array([0.0, -density*g, 0.0])
            # val = np.array([0.0, 0.0, density*g])
            val = np.array([0.0, density*g])
            return val
        return mass_map


class HyperElasticity_inv(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.

    def get_tensor_map(self):

        def psi(F):
            E = 2.842800754666559e+03
            nu = 0.48
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))

            # mu = 0.00635971757850452e6
            # kappa = 0.315865973065724e6 * 0.5

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
            density = 942.8238474  # kg/m³, breast tissue density
            g = 9.81  # m/s², gravitational acceleration
            # val = np.array([0.0, -density*g, 0.0])
            val = np.array([0.0, -density*g])
            # val = np.array([0.0, 0.0, -density*g])
            return val
        return mass_map


case_indicator = 166

# Vari_Flag
cut_loc             = 51
lambda_factor       = 1000
start_learning_rate = 1e-5
tolerance_last      = 1e-50
tolerance_relax     = 1e-50

# method_indicator = "gradually_increase_Known_mass_hmax_10_nu_48_location_51_strain_E_tilde_Lambda_100_i_0_include_tol_5e-4_Then_tol_2e-4_"
# method_indicator = "gradually_Known_hmax_20_nu_48_location_51_E_tilde_Lambda_1_tol_1e-3_lr_1e-6_"
method_indicator = (
    "gradually_Known_hmax_20_nu_48_location_"
    + str(cut_loc)
    + "_E_0_Lambda_"
    + str(lambda_factor)
    + "_tol_"
    + str(tolerance_last)
    + "_lr_"
    + str(start_learning_rate)
    + "_cap_inf_2D_"
)


# Specify mesh-related information (first-order hexahedron element).
ele_type = 'TET4'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
# Lx, Ly, Lz = 3., 3., 3.
# meshio_mesh = box_mesh_gmsh(Nx=3,
#                             Ny=3,
#                             Nz=3,
#                             Lx=Lx,
#                             Ly=Ly,
#                             Lz=Lz,
#                             data_dir=data_dir,
#                             ele_type=ele_type)
# mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# meshio_mesh = meshio.read("hemi_tet4_fine.inp")
# mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
# print(mesh.points)

########################################## MESH ############################################################
########################################## MESH ############################################################
########################################## MESH ############################################################
########################################## MESH ############################################################

# 1. Read the full mesh
mesh = meshio.read("test_hmax_20.msh")
# mesh = meshio.read("coarse_29_hmax_10.msh")


# 2. Inspect available cell types (optional debugging/insight)
#    This tells you what element types were read (triangle, tetra, etc.)
print("Cell types found:", mesh.cells_dict.keys())

# 3. Extract only the volume (3D) cells
volume_type = "tetra"      # or "hexahedron", "wedge", etc. depending on your mesh

if volume_type not in mesh.cells_dict:
    raise ValueError(f"No {volume_type} cells found in this mesh.")

# Filter out only tetra cells
volume_cells = mesh.cells_dict[volume_type]

# (Optionally) if you want to preserve cell data (like physical groups), do so as well:
volume_cell_data = {}
if "gmsh:physical" in mesh.cell_data_dict:
    volume_cell_data["gmsh:physical"] = {
        volume_type: mesh.cell_data_dict["gmsh:physical"].get(volume_type, [])
    }

# 4. Create a new mesh with only volume cells
volume_only_mesh = meshio.Mesh(
    points=mesh.points,
    cells=[(volume_type, volume_cells)],
    cell_data=volume_cell_data,  # optional, if you want the physical group data
)

volume_only_mesh.points = volume_only_mesh.points/1000

mesh = Mesh(volume_only_mesh.points, volume_only_mesh.cells_dict[volume_type])

cells = volume_only_mesh.cells_dict[volume_type]
########################################## MESH ############################################################
########################################## MESH ############################################################
########################################## MESH ############################################################
########################################## MESH ############################################################


# cells
# Define boundary locations.
def left(point):
    # print(np.isclose(point[0], Lx, atol=1e-5))
    return np.isclose(point[0], 0, atol=1e-5)

# def y_0(point):
#     # print(np.isclose(point[0], Lx, atol=1e-5))
#     return np.isclose(point[1], 0, atol=1e-5)
y_0_point = np.where(mesh.points[:,1]>=cut_loc/1000)[0]
    
def y_0(point, ind):
    return np.isin(ind, y_0_point) 


# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[y_0] *2, # Results in [left, left, left] + [right, right, right] 
                     [0, 1], # Components: [u_x, u_y, u_z] 
                     [zero_dirichlet_val] * 2]


problem_2 = HyperElasticity_inv(mesh,
                            vec=2,
                            dim=2,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info
)

sol_list_2,_ = solver(problem_2, solver_options={'petsc_solver': {}})

# Store the solution
u_sol_2 = sol_list_2[0]
print(u_sol_2)
part_indicator = '_intial_neg_g.vtk'
vtk_path_2 = os.path.join(data_dir, 'vtk', f"{method_indicator}{case_indicator}{part_indicator}")
os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
save_sol(problem_2.fes[0], u_sol_2, vtk_path_2)


mesh.points = mesh.points + onp.array(u_sol_2)

problem_2 = HyperElasticity(mesh,
                            vec=2,
                            dim=2,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info
)

sol_list_2,_ = solver(problem_2, solver_options={'petsc_solver': {}})

# Store the solution
u_sol_2 = sol_list_2[0]
print(u_sol_2)
part_indicator = '_intial_neg_g_to_positive_g.vtk'
vtk_path_2 = os.path.join(data_dir, 'vtk', f"{method_indicator}{case_indicator}{part_indicator}")
os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
save_sol(problem_2.fes[0], u_sol_2, vtk_path_2)


mesh.points = mesh.points + u_sol_2
# Create an instance of the problem.


observed_positions_2 = mesh.points 
observed_positions_2 = mesh.points
undeformed_coord = mesh.points
original_cood = observed_positions_2


mesh.points = onp.array(observed_positions_2)

# better initial guess
# mesh.points = onp.array((observed_positions_2 + undeformed_coord)/2)

density_init = 300
density_target = 1000 

params = np.array(original_cood) * 0 # 0.0001
tol = 1e-8
# fixed_bc_mask = np.abs(mesh.points[:,1] - 0) < tol
fixed_bc_mask = mesh.points[:, 1] >= cut_loc/1000
non_fixed_indices = np.where(~fixed_bc_mask)[0]
params = params[non_fixed_indices]

problem = HyperElasticity_opt(mesh, vec=2, dim=2, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, density=942.8238474, fixed_bc_mask=fixed_bc_mask)
# params = np.zeros_like(problem.mesh[0].points)
# params = np.ones_like(problem.mesh[0].points)


print(params)
print("HAHA")
# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem)
print("HOHO")


# sol_list = fwd_pred(params)


# print("sol_list")
# print(sol_list[0])

def test_fn(sol_list):
    print('test fun')
    data_error =  np.sum((((sol_list[0]+problem.mesh[0].points) - observed_positions_2))**2)/np.sum((observed_positions_2)**2) #/np.sum((observed_positions_2)**2)) #np.sum((sol_list[0] - u_sol_2)**2)
    return data_error

def constraint_fn(params):
    sol_list, _ = fwd_pred(params)
    constraint = problem.penalty_fn(sol_list, params)
    return constraint

def composed_fn(params):
    sol, _ = fwd_pred(params)
    data_error = test_fn(sol) #test_fn(fwd_pred(params))
    constraint = problem.penalty_fn(sol, params)
    

    return data_error + constraint * lambda_factor


# def composed_fn(params):
#     sol, _ = fwd_pred(params)
#     data_error = test_fn(sol) #test_fn(fwd_pred(params))
#     # constraint = problem.penalty_fn(sol, params)
#     return data_error

#Vari_Flag

d_coord= jax.grad(composed_fn)(params)


params = np.array(original_cood) * 0
params = params[non_fixed_indices]
observed_positions_2_non_fixed = observed_positions_2[non_fixed_indices]

learning_rate = start_learning_rate
max_iterations = 5000


# current_mesh_dummy = mesh
current_mesh = mesh
relax_flag = 0
# original_density_arr = [50 ,100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 942.8238474]
original_density_arr = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 942.8238474]
original_density_arr = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 942.8238474]
# original_density_arr = [942.8238474]
# original_density_arr = [500, 550, 600, 650, 700, 750, 800, 850, 900, 942.8238474]
# original_density_arr = onp.arange(50, 943, 1)
# original_density_arr = [200, 400, 600, 800, 942.8238474]
# original_density_arr = [200, 600, 800, 1000]
density_arr = original_density_arr
step_roll_back_flag = 0
negative_J_count = 0
fail_count = 0
last_5_costs = []
cost_history = []  # reinitialize if you want per-density cost history
problem = HyperElasticity_opt(current_mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, density=density_arr[0], fixed_bc_mask=fixed_bc_mask)
fail_count_for_count = 0
fwd_pred = ad_wrapper(problem)
start = time.time()
sol_list = fwd_pred(params)
u_sol_opt = sol_list[0]
part_indicator = '_before_opt.vtk'
vtk_path_opt = os.path.join(data_dir, 'vtk', f"{method_indicator}{case_indicator}{part_indicator}")
os.makedirs(os.path.dirname(vtk_path_opt), exist_ok=True)
save_sol(problem.fes[0], u_sol_opt, vtk_path_opt)

# Define the memory threshold (e.g., 80% of total memory)
memory_threshold = 70.0  # In percentage -- 19456

print("//////////////////////////")
print("//////////////////////////")
print("//////////////////////////")
print("//////////////////////////")
print("OPT Start")
print("//////////////////////////")
print("//////////////////////////")
print("//////////////////////////")
print("//////////////////////////")

def tet_signed_volume(v0, v1, v2, v3):
    # Compute the signed volume of a tetrahedron.
    return onp.linalg.det(onp.column_stack((v1 - v0, v2 - v0, v3 - v0))) / 6.0

def refine_interval(density_arr, d_fail, original_density_arr, fail_count):
    """
    Refine the interval (between two original endpoints) in which d_fail lies.
    
    Rather than partially preserving the old array, this version replaces the entire
    interval [d_low, d_high] with a new, uniformly spaced set of points.
    
    The number of refined points is given by:
         n_point = 2**(fail_count) + 1
         
    Examples (assuming original_density_arr = [200, 400, 600, 800, 1000]):
      - If density_arr is [200,400,600,800,1000] and d_fail is 800 with fail_count = 1,
        then new_points = [600, 700, 800] and the updated density_arr becomes 
        [200, 400, 600, 700, 800, 1000].
        
      - If later density_arr is [200,400,600,700,800,1000] and d_fail is 700 with fail_count = 2,
        then new_points = [600, 650, 700, 750, 800] and the updated density_arr becomes 
        [200, 400, 600, 650, 700, 750, 800, 1000].
        
      - And so on.
    """
    # Identify the original endpoints that bracket d_fail.
    for i in range(len(original_density_arr) - 1):
        if original_density_arr[i] <= d_fail <= original_density_arr[i+1]:
            d_low = original_density_arr[i]
            d_high = original_density_arr[i+1]
            break
    else:
        raise ValueError(f"d_fail {d_fail} is not between any endpoints in original_density_arr.")
    
    # Find the indices in density_arr corresponding to d_low and d_high.
    try:
        start_index = density_arr.index(d_low)
        end_index = density_arr.index(d_high)
    except ValueError:
        raise ValueError("The endpoints {} or {} are not in the current density_arr."
                         .format(d_low, d_high))
    
    # Calculate the number of points for uniform refinement.
    n_point = 2**(fail_count) + 1  # For fail_count=1, n_point=3; for 2, n_point=5; etc.
    # Generate the new uniformly spaced points.
    new_points = onp.linspace(d_low, d_high, n_point).tolist()
    
    # Replace the entire interval [d_low, d_high] with the refined points.
    new_density_arr = density_arr[:start_index] + new_points + density_arr[end_index+1:]
    return new_density_arr

# Open CSV file for writing cost history in real-time
part_indicator = '_cost_history_learning_rate.csv'
csv_file_path = f"{method_indicator}{case_indicator}{part_indicator}"

mesh_for_cal = mesh

with open(csv_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Cost", "learning_rate", "Density", "penalty", "avg_volume", "min_volume",'Negative_Jaco_Flag'])  # Write header

    # We use a while loop over the density list so that we can insert new values if needed.
    i = 0  # index for density_arr
    i_vtk_count = 0
    while i < len(density_arr):
        cost_history = []
        current_density = density_arr[i]
        problem.density = current_density
        print(f"====================\nCurrent density = {current_density}\n====================")
        
        # If we are not coming from a rollback, then store the current parameters 
        # (and also the previous density for the purpose of computing the new density if needed).
        # (Note: for i == 0 there is no previous density, so we simply keep the current one.)
        if step_roll_back_flag == 0:
            prev_param = params.copy()  # make sure to copy if params is a mutable array

        # Reset the rollback flag for this density level.
        step_roll_back_flag = 0

        # -------------------------------
        # Inner optimization loop for current density
        # -------------------------------
        
        for iteration in range(max_iterations):
            
            try:    
                # Compute gradient and normalize if necessary.
                d_coord = jax.grad(composed_fn)(params)
                
                grad_norm = np.linalg.norm(d_coord)

                if grad_norm > 1e-8:    
                    d_coord = d_coord / grad_norm

                # Update parameters
                params = params - learning_rate * d_coord
                # print('a')
                # Solve FEM problem


                sol_list, _ = fwd_pred(params)
                # print('b')
                
                BB = problem.penalty_fn(sol_list, params)

                penalty = BB * lambda_factor
                # Vari_Flag
                GGG = 0
            except:
                GGG = 1
                # penalty = 0


            # if sol_list is None:
            if GGG == 1:

                # Roll back this step: undo the last parameter update
                params = params + learning_rate * d_coord
                # Reduce learning rate to help convergence next time
                learning_rate *= 0.5
                negative_J_count += 1

                print('===================================================================================================================')
                print('===================================================================================================================')
                print('===================================================================================================================')
                print('===================================================================================================================')
                print("Encountered negative J (or invalid FEM result)!")
                print(f"Iteration {iteration}: learning_rate = {learning_rate}, current_density = {current_density}, penalty = {penalty}")
                print('===================================================================================================================')
                print('===================================================================================================================')
                print('===================================================================================================================')
                print('===================================================================================================================')

                mesh_for_cal.points[non_fixed_indices] = current_mesh.points[non_fixed_indices] + params
                volumes = [tet_signed_volume(*mesh_for_cal.points[cell]) for cell in cells]
                avg_volume = onp.mean(volumes)
                min_volume = onp.min(volumes)
                print("Average volume:", avg_volume)
                print("Minimum volume:", min_volume)

                writer.writerow([iteration, 0, learning_rate, current_density,penalty, avg_volume, min_volume, GGG])  # Save immediately
                file.flush()  # Ensure data is written to disk
            else:
                u_sol = np.zeros_like(sol_list[0])

                np_points_temp = np.array(problem.mesh[0].points)
                reconstructed_param_temp = np.zeros_like(np_points_temp)
                reconstructed_param_temp = reconstructed_param_temp.at[non_fixed_indices].set(params)
                part_indicator = '_ain_'
                vtk_path = os.path.join(data_dir, 'vtk', f"{method_indicator}/{method_indicator}{case_indicator}{part_indicator}{i_vtk_count}.vtk")
                os.makedirs(os.path.dirname(vtk_path), exist_ok=True)
                # save_sol(problem.fes[0], u_sol, vtk_path)
                mesh_temp = problem.mesh[0]
                mesh_temp.points = mesh_temp.points + onp.array(reconstructed_param_temp)
                save_sol(mesh_temp, u_sol, vtk_path)
                i_vtk_count = i_vtk_count + 1

                negative_J_count = 0
                # Optionally increase the learning rate slowly
                if learning_rate < start_learning_rate:
                    learning_rate *= 1.05
                else:
                    learning_rate = start_learning_rate

                last_3 = cost_history[-3:]
                if len(cost_history) >= 3:
                    if onp.all(onp.diff(last_3) > 0):
                        learning_rate *= 0.5
                        print(f"Costs rising → reducing lr to {learning_rate:.3e}")



                # Compute cost for convergence checking
                current_cost = np.sum((((sol_list[0] + problem.mesh[0].points) - observed_positions_2))**2)/np.sum((observed_positions_2)**2)
                cost_history.append(current_cost)

                penalty_term = penalty
                # capped_penalty = np.minimum(1*penalty, 100 * current_cost)
                # jax.debug.print("capped_penalty: {}", capped_penalty)
                print("penalty_term:", penalty_term)
                # capped_penalty = np.minimum(penalty, 100 * current_cost)

                current_cost = current_cost
                mesh_for_cal.points[non_fixed_indices] 
                current_mesh.points[non_fixed_indices] + params
                volumes = [tet_signed_volume(*mesh_for_cal.points[cell]) for cell in cells]
                avg_volume = onp.mean(volumes)
                min_volume = onp.min(volumes)
                print("Average volume:", avg_volume)
                print("Minimum volume:", min_volume)

                writer.writerow([iteration, current_cost, learning_rate, current_density, penalty_term, avg_volume, min_volume, GGG])  # Save immediately
                file.flush()  # Ensure data is written to disk
                print('===================================================================================================================')
                print('===================================================================================================================')
                print('===================================================================================================================')
                print('===================================================================================================================')
            
                print(f"Iteration {iteration}: Cost = {current_cost}, learning_rate = {learning_rate}, current_density = {current_density}, penalty = {penalty_term}")

                print('===================================================================================================================')
                print('===================================================================================================================')
                print('===================================================================================================================')
                print('===================================================================================================================')

                # Convergence check (you can adjust this condition as needed)
                if current_density == density_arr[-1]:
                    if iteration >= 0 and np.abs(current_cost) < tolerance_last: 
                        print("Converged at target density!")
                        break
                elif iteration >= 0 and np.abs(current_cost) < tolerance_relax:
                    if current_density in original_density_arr:
                        fail_count = 0
                    print("Converged at this density!")
                    break

                # Check if we have a plateau or repeated negative J events
                # if iteration > 6:
                #     last_5_costs = [row[1] for row in cost_history[-5:]]
                #     last_5_costs = np.array(last_5_costs)
                #     if np.std(last_5_cospenalty_fnts) < 1e-5:
                #         print("Optimization stalled or negative J occurred several times!")
                #         step_roll_back_flag = 1
                #         break

            if negative_J_count > 7:
                print("Optimization stalled or negative J occurred several times!")
                step_roll_back_flag = 1
                negative_J_count = 0
                break
            
            # # Check system memory usage
            memory_info = psutil.virtual_memory()
            if memory_info.percent > memory_threshold:
                print(f"Memory usage exceeded {memory_threshold}%. Clearing caches...")
                jax.clear_caches()
                gc.collect()


        # -------------------------------
        # End of inner optimization loop
        # -------------------------------

        if step_roll_back_flag:
            fail_count = fail_count + 1
            fail_count_for_count = fail_count_for_count + 1
            # Roll back parameters to the last successful state
            params = prev_param.copy()
            # Compute new density to try: halfway between the previous successful density and the current one.
            new_density = refine_interval(density_arr, current_density, original_density_arr, fail_count)
            i = new_density.index(current_density) - 1
            # step_roll_back_flag = 0
            learning_rate = start_learning_rate
            density_arr = new_density
            continue  # do not advance i
        else:
            # Optimization at the current density was successful.
            print(f"Density {current_density} optimization successful.")
            i += 1  # move on to the next density level

    print("Optimization over all density levels completed.")


end = time.time()
print('Run Time:')
print((end - start)/60)


updated_mesh_points = onp.copy(current_mesh.points)
updated_mesh_points[non_fixed_indices] = updated_mesh_points[non_fixed_indices] + params

# current_mesh = current_mesh_dummy
current_mesh.points = updated_mesh_points


problem.mesh[0].points = updated_mesh_points
    # Step 5: Solve the FEM problem with the updated geometry
params = params * 0
sol_list = fwd_pred(params)

u_sol_opt = sol_list[0]
part_indicator = '_after_optimization.vtu'
vtk_path_opt = os.path.join(data_dir, 'vtk', f"{method_indicator}{case_indicator}{part_indicator}")
os.makedirs(os.path.dirname(vtk_path_opt), exist_ok=True)
save_sol(problem.fes[0], u_sol_opt, vtk_path_opt)

# Convert to pandas DataFrame
undeformed_df = pd.DataFrame(undeformed_coord, columns=["X", "Y", "Z"])
initial_gauss_df = pd.DataFrame(observed_positions_2, columns=["X", "Y", "Z"])
optimized_df = pd.DataFrame(current_mesh.points, columns=["X", "Y", "Z"])


# Save to CSV files
addr = '/home/gusdh/jax-fem/demos/hyperelasticity/jax-fem/demos/prac_hy/data/By_JAX/'


part_indicator = '_undeformed_coordinates.csv'
undeformed_df.to_csv(f"{addr}{method_indicator}{case_indicator}{part_indicator}", index=False)

part_indicator = '_initial_gauss.csv'
initial_gauss_df.to_csv(f"{addr}{method_indicator}{case_indicator}{part_indicator}", index=False)

part_indicator = '_optimized_coordinates.csv'
optimized_df.to_csv(f"{addr}{method_indicator}{case_indicator}{part_indicator}", index=False)

print('fail_count_for_count')
print(fail_count_for_count)