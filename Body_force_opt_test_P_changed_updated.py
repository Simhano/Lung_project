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


# Define constitutive relationship.
class HyperElasticity_opt(Problem):
    def __init__(self, *args,density=2.0, **kwargs):
        self.density = density  # Make internal pressure variable
        super().__init__(*args, **kwargs)

    def custom_init(self):
        self.fe = self.fes[0]
        tol = 1e-8
        self.fixed_bc_mask = np.abs(mesh.points[:, 1] - 0) < tol  # Use NumPy here
        self.non_fixed_indices = np.where(~self.fixed_bc_mask)[0]
        self.np_points = np.array(self.mesh[0].points)
        self.param_flag = 0
    def get_tensor_map(self):

        def psi(F_tilde):
            E = 5000.
            nu = 0.4
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
            return energy

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
            val = np.array([0.0, 0.0, density*g])
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
            E = 5000.
            nu = 0.4
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
            density = 1000  # kg/m³, breast tissue density
            g = 9.81  # m/s², gravitational acceleration
            # val = np.array([0.0, -density*g, 0.0])
            val = np.array([0.0, 0.0, density*g])
            return val
        return mass_map



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


meshio_mesh = meshio.read("hemi_tet4_fine.inp")
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
# print(mesh.points)


# Define boundary locations.
def left(point):
    # print(np.isclose(point[0], Lx, atol=1e-5))
    return np.isclose(point[0], 0, atol=1e-5)

def y_0(point):
    # print(np.isclose(point[0], Lx, atol=1e-5))
    return np.isclose(point[1], 0, atol=1e-5)

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
vtk_path_2 = os.path.join(data_dir, 'vtk', 'u_undeformed_config_body_hemi_non_post_itni.vtu')
os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
save_sol(problem_2.fes[0], u_sol_2, vtk_path_2)

# Create an instance of the problem.





# mesh_points_modi = mesh.points 8.79406 -3.2669
# scale_d = 1.
# original_cood = np.copy(mesh.points) * 1.1


observed_positions_2 = mesh.points + u_sol_2
undeformed_coord = mesh.points
original_cood = observed_positions_2

mesh.points = onp.array(observed_positions_2)

# better initial guess
# mesh.points = onp.array((observed_positions_2 + undeformed_coord)/2)

density_init = 300
density_target = 1000 

problem = HyperElasticity_opt(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, density=1000)
# params = np.zeros_like(problem.mesh[0].points)
# params = np.ones_like(problem.mesh[0].points)



params = np.array(original_cood) * 0 # 0.0001
tol = 1e-8
fixed_bc_mask = np.abs(mesh.points[:,1] - 0) < tol
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
    return np.sum((((sol_list[0]+problem.mesh[0].points) - observed_positions_2))**2)/np.sum((observed_positions_2)**2)  #/np.sum((observed_positions_2)**2)) #np.sum((sol_list[0] - u_sol_2)**2)
    #Set parameter without fixed nodes.
    #Normalize
     
def composed_fn(params):
    Sol = fwd_pred(params)
    if Sol == None:
        return None
    else:
        return np.sum(test_fn(Sol)) #test_fn(fwd_pred(params))


d_coord= jax.grad(composed_fn)(params)


print("d_coord")
print(d_coord.shape)
print(d_coord)


params = np.array(original_cood) * 0
params = params[non_fixed_indices]
observed_positions_2_non_fixed = observed_positions_2[non_fixed_indices]
start_learning_rate = 0.05
learning_rate = start_learning_rate
max_iterations = 500
tolerance_last = 1e-4
tolerance_relax = 1e-3
# current_mesh_dummy = mesh
current_mesh = mesh
relax_flag = 0
density_arr = [200, 300, 400, 500, 600, 700, 750, 800, 850, 900, 950, 1000]
original_density_arr = [200, 400, 600, 800, 1000]
density_arr = original_density_arr
step_roll_back_flag = 0
negative_J_count = 0
fail_count = 0
last_5_costs = []
cost_history = []  # reinitialize if you want per-density cost history
problem = HyperElasticity_opt(current_mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info, density=density_arr[-1])
fail_count_for_count = 0
fwd_pred = ad_wrapper(problem)
start = time.time()
sol_list = fwd_pred(params)
u_sol_opt = sol_list[0]
vtk_path_opt = os.path.join(data_dir, 'vtk', 'u_pressure_opt_with_JAX_body_before_opt_hemi_non_post_itni.vtu')
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


# We use a while loop over the density list so that we can insert new values if needed.
i = 0  # index for density_arr

while i < len(density_arr):
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

        # Compute gradient and normalize if necessary.
        d_coord = jax.grad(composed_fn)(params)
        grad_norm = np.linalg.norm(d_coord)
        if grad_norm > 1e-8:
            d_coord = d_coord / grad_norm

        # Update parameters
        params = params - learning_rate * d_coord

        # Solve FEM problem
        sol_list = fwd_pred(params)
        if sol_list is None:

            # Roll back this step: undo the last parameter update
            params = params + learning_rate * d_coord
            # Reduce learning rate to help convergence next time
            learning_rate *= 0.5
            negative_J_count += 1
            cost_history.append([iteration, 0, learning_rate, current_density])
            print('===================================================================================================================')
            print('===================================================================================================================')
            print('===================================================================================================================')
            print('===================================================================================================================')
            print("Encountered negative J (or invalid FEM result)!")
            print(f"Iteration {iteration}: learning_rate = {learning_rate}, current_density = {current_density}")
            print('===================================================================================================================')
            print('===================================================================================================================')
            print('===================================================================================================================')
            print('===================================================================================================================')
        else:
            negative_J_count = 0
            # Optionally increase the learning rate slowly
            learning_rate *= 1.05

            # Compute cost for convergence checking
            current_cost = np.sum((((sol_list[0] + problem.mesh[0].points) - observed_positions_2))**2)
            cost_history.append([iteration, current_cost,learning_rate, current_density])
            print('===================================================================================================================')
            print('===================================================================================================================')
            print('===================================================================================================================')
            print('===================================================================================================================')
        
            print(f"Iteration {iteration}: Cost = {current_cost}, learning_rate = {learning_rate}, current_density = {current_density}")

            print('===================================================================================================================')
            print('===================================================================================================================')
            print('===================================================================================================================')
            print('===================================================================================================================')

            # Convergence check (you can adjust this condition as needed)
            if current_density == density_arr[-1]:
                if iteration > 0 and np.abs(current_cost) < tolerance_last: 
                    print("Converged at target density!")
                    break
            elif iteration > 0 and np.abs(current_cost) < tolerance_relax:
                if current_density in original_density_arr:
                    fail_count = 0
                print("Converged at this density!")
                break

            # Check if we have a plateau or repeated negative J events
            if iteration > 6:
                last_5_costs = [row[1] for row in cost_history[-5:]]
                last_5_costs = np.array(last_5_costs)
                if np.std(last_5_costs) < 1e-5:
                    print("Optimization stalled or negative J occurred several times!")
                    step_roll_back_flag = 1
                    break

        if negative_J_count > 4:
            print("Optimization stalled or negative J occurred several times!")
            step_roll_back_flag = 1
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
        step_roll_back_flag = 0
        learning_rate = 0.025
        density_arr = new_density
        continue  # do not advance i
    else:
        # Optimization at the current density was successful.
        print(f"Density {current_density} optimization successful.")
        i += 1  # move on to the next density level

print("Optimization over all density levels completed.")





# Save cost history to a CSV file
with open("cost_history_learning_rate_body_hemi_non_post_itni.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Cost", 'learning_rate' , 'Density'])  # Write header
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
sol_list = fwd_pred(params)

u_sol_opt = sol_list[0]
vtk_path_opt = os.path.join(data_dir, 'vtk', 'u_pressure_opt_with_JAX_body_hemi_non_post_itni_param_up.vtu')
os.makedirs(os.path.dirname(vtk_path_opt), exist_ok=True)
save_sol(problem.fes[0], u_sol_opt, vtk_path_opt)

# Convert to pandas DataFrame
undeformed_df = pd.DataFrame(undeformed_coord, columns=["X", "Y", "Z"])
initial_gauss_df = pd.DataFrame(observed_positions_2, columns=["X", "Y", "Z"])
optimized_df = pd.DataFrame(current_mesh.points, columns=["X", "Y", "Z"])


# Save to CSV files
undeformed_df.to_csv("/home/gusdh/jax-fem/demos/hyperelasticity/jax-fem/demos/prac_hy/data/By_SciPy/undeformed_coordinates_JAX_body_hemi_non_post_itni.csv", index=False)
initial_gauss_df.to_csv("/home/gusdh/jax-fem/demos/hyperelasticity/jax-fem/demos/prac_hy/data/By_SciPy/initial_gauss_coordinates_body_hemi_non_post_itni.csv", index=False)
optimized_df.to_csv("/home/gusdh/jax-fem/demos/hyperelasticity/jax-fem/demos/prac_hy/data/By_SciPy/optimized_coordinates_body_hemi_non_post_itni.csv", index=False)

print('fail_count_for_count')
print(fail_count_for_count)