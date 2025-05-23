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

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh


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
            nu = 0.46
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
            # threshold = 0.5
            # penalty_weight = 1e3

            # Using a smooth approximation: softplus gives a smooth penalty
            # penalty = penalty_weight * jax.nn.softplus(threshold - J)
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
            val = np.array([0.0, density*g, 0.0])
            # jax.debug.print("density: {}", density)
            return val
        return mass_map

    def set_params(self, params):
        # self.X_0 = self.mesh[0].points + params
        self.np_points = np.array(self.mesh[0].points)
        reconstructed_param = np.zeros_like(self.np_points)
        reconstructed_param = reconstructed_param.at[self.non_fixed_indices].set(params)
        # reconstructed_param = reconstructed_param.at[~fixed_bc_mask].set(params)

        self.X_0 = self.np_points + reconstructed_param
        # jax.debug.print("X_0: {}", self.X_0)
        # self.params
        self.params = reconstructed_param

        # print('')

        self.__post_init__()

        # self.param_flag = self.param_flag + 1
        # jax.debug.print("self.param_flag: {}", self.param_flag)


        
    

class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.

    def get_tensor_map(self):

        def psi(F):
            E = 2.842800754666559e+03
            nu = 0.46
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))

            # jax.debug.print("mu: {}", mu)
            # jax.debug.print("kappa: {}", kappa)
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
            val = np.array([0.0, density*g, 0.0])
            # val = np.array([0.0, 0.0, density*g])
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
            nu = 0.46
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
            val = np.array([0.0, -density*g, 0.0])
            # val = np.array([0.0, 0.0, -density*g])
            return val
        return mass_map


case_indicator = 166
method_indicator = "negative_force_inital_hmax_20_nu_46_location_51_"



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
# print(volume_only_mesh.points)
# print(volume_only_mesh.cells_dict["tetra"])

volume_only_mesh.points = volume_only_mesh.points/1000

mesh = Mesh(volume_only_mesh.points, volume_only_mesh.cells_dict[volume_type])

########################################## MESH ############################################################
########################################## MESH ############################################################
########################################## MESH ############################################################
########################################## MESH ############################################################








# Define boundary locations.
def left(point):
    # print(np.isclose(point[0], Lx, atol=1e-5))
    return np.isclose(point[0], 0, atol=1e-5)

# def y_0(point):
#     # print(np.isclose(point[0], Lx, atol=1e-5))
#     return np.isclose(point[1], 0, atol=1e-5)

y_0_point = np.where(mesh.points[:,1]>=51/1000)[0]
    
def y_0(point, ind):
    return np.isin(ind, y_0_point) 

# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[y_0] *3, # Results in [left, left, left] + [right, right, right] 
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
part_indicator = '_intial_forward.vtk'
vtk_path_2 = os.path.join(data_dir, 'vtk', f"{method_indicator}{case_indicator}{part_indicator}")
os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
save_sol(problem_2.fes[0], u_sol_2, vtk_path_2)

# Create an instance of the problem.





# mesh_points_modi = mesh.points 8.79406 -3.2669
# scale_d = 1.
# original_cood = np.copy(mesh.points) * 1.1


observed_positions_2 = mesh.points
undeformed_coord = mesh.points
original_cood = observed_positions_2

mesh.points = onp.array(observed_positions_2)
problem_inv = HyperElasticity_inv(mesh,
                            vec=3,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info
)

sol_list_inv = solver(problem_inv, solver_options={'petsc_solver': {}})
u_sol_inv = sol_list_inv[0]
part_indicator = '_intial_guess.vtk'
vtk_path_2 = os.path.join(data_dir, 'vtk', f"{method_indicator}{case_indicator}{part_indicator}")
os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
save_sol(problem_inv.fes[0], u_sol_inv, vtk_path_2)

init_guess = mesh.points + u_sol_inv
mesh.points = onp.array(init_guess)

# better initial guess
# mesh.points = onp.array((observed_positions_2 + undeformed_coord)/2)

density_init = 300
density_target = 1000 

params = np.array(original_cood) * 0 # 0.0001
tol = 1e-8
# fixed_bc_mask = np.abs(mesh.points[:,1] - 0) < tol
fixed_bc_mask = mesh.points[:, 1] >= 51/1000
non_fixed_indices = np.where(~fixed_bc_mask)[0]
params = params[non_fixed_indices]

problem = HyperElasticity_opt(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, density=1000, fixed_bc_mask=fixed_bc_mask)
# params = np.zeros_like(problem.mesh[0].points)
# params = np.ones_like(problem.mesh[0].points)

print(params)
print("HAHA")
# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem)
print("HOHO")
sol_list = fwd_pred(params)
# print("sol_list")
# print(sol_list[0])

def test_fn(sol_list, penalty):
    print('test fun')
    
    # jax.debug.print("cost func: {}", np.sum((sol_list[0] - u_sol_2)**2))
    data_error =  np.sum((((sol_list[0]+problem.mesh[0].points) - observed_positions_2))**2)/np.sum((observed_positions_2)**2) #/np.sum((observed_positions_2)**2)) #np.sum((sol_list[0] - u_sol_2)**2)
    # capped_penalty = np.minimum(1*penalty, 100 * data_error)
    capped_penalty = 10*penalty
    # jax.debug.print("capped_penalty: {}", capped_penalty)
    return data_error + capped_penalty
    #Set parameter without fixed nodes.
    #Normalize
     
def composed_fn(params):
    Sol, penalty = fwd_pred(params)
    return np.sum(test_fn(Sol, penalty)) #test_fn(fwd_pred(params))


d_coord= jax.grad(composed_fn)(params)


print("d_coord")
print(d_coord.shape)
print(d_coord)


params = np.array(original_cood) * 0
params = params[non_fixed_indices]
observed_positions_2_non_fixed = observed_positions_2[non_fixed_indices]
start_learning_rate = 0.001
learning_rate = start_learning_rate
max_iterations = 500
tolerance = 1e-3
# current_mesh_dummy = mesh
current_mesh = mesh
relax_flag = 0
cost_history = []
density_init = 300
density_target = 1000 
density_arr = [200, 300, 400, 500, 600, 700, 750, 800, 850, 900, 950, 1000]
# density_arr = [800, 850, 942.8238474]
density_arr = [942.8238474]
density_gap = 200
step_roll_back_flag = 0
negative_J_count = 0
problem = HyperElasticity_opt(current_mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, density=density_init, fixed_bc_mask=fixed_bc_mask)

fwd_pred = ad_wrapper(problem)
start = time.time()
Sol, penalty = fwd_pred(params)
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
i_vtk_count = 0

for BC_i in range(len(density_arr)):
    problem.density = density_arr[BC_i]

    for iteration in range(max_iterations):

        # params = np.zeros_like(original_cood)
        # params = params * 0

        # Step 1: Compute gradient
        try:
            # Compute gradient and normalize if necessary.
            d_coord = jax.grad(composed_fn)(params)
            grad_norm = np.linalg.norm(d_coord)
            if grad_norm > 1e-8:
                d_coord = d_coord / grad_norm

            # Update parameters
            params = params - learning_rate * d_coord

            # Solve FEM problem
 
            Sol, penalty = fwd_pred(params)
            GGG = 0
        except:
            GGG = 1

        if GGG == 1:
            print("J < 0!!!")
            print("J < 0!!!")
            # print("J < 0!!!")
            # print("J < 0!!!")
            # print("J < 0!!!")

            params = params + learning_rate * d_coord
            learning_rate = learning_rate * 0.5
            print('learning_rate')
            print(learning_rate)
            print('density_arr[BC_i]')
            print(density_arr[BC_i])
            negative_J_count = negative_J_count + 1

            # relax_flag = relax_flag + 1

            # if relax_flag > 3:
            #     # params = params + observed_positions_2_non_fixed * 0.01
            #     print('Relax Relax Relax Relax')
            #     print('Relax Relax Relax Relax')
            #     print('Relax Relax Relax Relax')
            #     print('Relax Relax Relax Relax')

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
            learning_rate = learning_rate * 1.05

        #####
            # u_sol_opt = sol_list[0]
            # vtk_path_opt = os.path.join(data_dir, 'vtk', 'u_pressure_opt_with_JAX_body_hemi_during_Iteration.vtu')
            # os.makedirs(os.path.dirname(vtk_path_opt), exist_ok=True)
            # save_sol(problem.fes[0], u_sol_opt, vtk_path_opt)
        #######
            # Step 6: Compute the cost for convergence
            current_cost = np.sum((((sol_list[0]+problem.mesh[0].points) - observed_positions_2))**2)
            cost_history.append([iteration, current_cost])  # Save itePETScration and cost
            # print("//////////////////////////")
            # print("//////////////////////////")
            print("//////////////////////////")
            print("//////////////////////////")
            print(f"Iteration {iteration}: Cost = {current_cost}")
            # print("//////////////////////////")
            # print("//////////////////////////")
            print("//////////////////////////")
            print("//////////////////////////")
            print('learning_rate')
            print(learning_rate)
            print("density")
            print(density_arr[BC_i])
            print("//////////////////////////")
            print("//////////////////////////")
            # print("//////////////////////////")
            # print("//////////////////////////")
            # if current_cost < 2:
            #     learning_rate = 0.01

                # Check for convergence
            # if iteration > 0 and np.abs(prev_cost - current_cost) < tolerance:
            if iteration > 0 and np.abs(current_cost) < tolerance:
                print("Converged!")
                break

            # prev_cost = current_cost

            #memory_threshold = 70.0  # In percentage -- 19456


        # # Check system memory usage
        memory_info = psutil.virtual_memory()
        if memory_info.percent > memory_threshold:
            print(f"Memory usage exceeded {memory_threshold}%. Clearing caches...")
            jax.clear_caches()
            gc.collect()




# Save cost history to a CSV file
part_indicator = '_cost_history_learning_rate.csv'

with open(f"{method_indicator}{case_indicator}{part_indicator}", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Cost"])  # Write header
    writer.writerows(cost_history)         # Write data


end = time.time()
print('Run Time:')
print((end - start)/60)


updated_mesh_points = onp.copy(current_mesh.points)
updated_mesh_points[non_fixed_indices] = updated_mesh_points[non_fixed_indices] + params

# current_mesh = current_mesh_dummy
current_mesh.points = updated_mesh_points

# del problem, fwd_pred, sol_list # Replace with variables no longer needed
# gc.collect()

problem.mesh[0].points = updated_mesh_points
    # Step 5: Solve the FEM problem with the updated geometry
params = params * 0
Sol, penalty = fwd_pred(params)

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
