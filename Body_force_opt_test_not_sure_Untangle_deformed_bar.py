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



# Define constitutive relationship.
class HyperElasticity_opt(Problem):

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
            # jax.debug.print("jacobian_det: {}", J)

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
            density = 0.01  # kg/m³, breast tissue density
            g = 9.81  # m/s², gravitational acceleration
            # val = np.array([0.0, -density*g, 0.0])
            val = np.array([0.01, 0.0, 0.0])
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

        self.param_flag = self.param_flag + 1
        jax.debug.print("self.param_flag: {}", self.param_flag)


        




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


meshio_mesh = meshio.read("deformed_bar.inp")
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
# print(mesh.points)
points = meshio_mesh.points
cells = meshio_mesh.cells_dict[cell_type]

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




# Create an instance of the problem.





# mesh_points_modi = mesh.points 8.79406 -3.2669
# scale_d = 1.
# original_cood = np.copy(mesh.points) * 1.1


observed_positions_2 = mesh.points
original_cood = observed_positions_2

mesh.points = onp.array(observed_positions_2)

# better initial guess
# mesh.points = onp.array((observed_positions_2 + undeformed_coord)/2)

problem = HyperElasticity_opt(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
# params = np.zeros_like(problem.mesh[0].points)
# params = np.ones_like(problem.mesh[0].points)



params = np.array(original_cood) * 0 # 0.0001
tol = 1e-8
fixed_bc_mask = np.abs(mesh.points[:,1] - 0) < tol
non_fixed_indices = np.where(~fixed_bc_mask)[0]
params = params[non_fixed_indices]



def tet_signed_volume(v0, v1, v2, v3):
    # Compute the signed volume of a tetrahedron.
    return onp.linalg.det(onp.column_stack((v1 - v0, v2 - v0, v3 - v0))) / 6.0

def global_untangle_tet_mesh(points, cells, fixed_mask, vol_threshold=1e-6, scale_factor=1.1, max_iters=50):
    """
    Untangles a tetrahedral mesh while keeping fixed nodes (specified by fixed_mask) unchanged.
    Additionally, it logs which cell (element) is modified in each iteration.
    
    Parameters:
      points: (n_points, 3) array of node coordinates.
      cells: (n_cells, 4) array of tetrahedral connectivity (indices into points).
      fixed_mask: Boolean array of length n_points. True means the node is fixed.
      vol_threshold: Minimum acceptable element volume.
      scale_factor: Factor to expand a bad element.
      max_iters: Maximum iterations to attempt untangling.
      
    Returns:
      new_points: The updated node coordinates.
      changes_per_iteration: Dictionary mapping iteration number to list of changed cell indices.
    """
    new_points = points.copy()
    changes_per_iteration = {}
    
    for iteration in range(max_iters):
        # Prepare accumulators for corrections and update counts per node.
        corrections = onp.zeros_like(new_points)
        counts = onp.zeros(new_points.shape[0], dtype=int)
        
        # List to record cell indices that are being corrected this iteration.
        changed_cells = []
        
        # Loop over each tetrahedral element, using enumerate to get its index.
        for cell_index, cell in enumerate(cells):
            v0, v1, v2, v3 = new_points[cell]
            vol = tet_signed_volume(v0, v1, v2, v3)
            if vol < vol_threshold:
                changed_cells.append(cell_index)
                centroid = (v0 + v1 + v2 + v3) / 4.0
                # Update only the free nodes of this element.
                for idx in cell:
                    if not fixed_mask[idx]:
                        delta = (new_points[idx] - centroid) * (scale_factor - 1.0)
                        corrections[idx] += delta
                        counts[idx] += 1
        
        # Log the changed cells for this iteration.
        if changed_cells:
            changes_per_iteration[iteration] = changed_cells
            print(f"Iteration {iteration+1}: Changed elements (indices): {changed_cells}")
        else:
            print(f"Iteration {iteration+1}: No changed elements.")
            print(f"Mesh untangled after {iteration+1} iterations.")
            break
        
        # Apply the average correction only to free nodes.
        for i in range(len(new_points)):
            if counts[i] > 0 and not fixed_mask[i]:
                new_points[i] += corrections[i] / counts[i]
    
    else:
        print("Warning: Maximum iterations reached; some bad elements may remain.")
    
    return new_points, changes_per_iteration




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
learning_rate = 0.5 #0.8 #0.0005 fail
max_iterations = 300
tolerance = 0.7
# current_mesh_dummy = mesh
current_mesh = mesh
relax_flag = 0
cost_history = []

problem = HyperElasticity_opt(current_mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info)
fwd_pred = ad_wrapper(problem)
start = time.time()
sol_list = fwd_pred(params)
u_sol_opt = sol_list[0]
vtk_path_opt = os.path.join(data_dir, 'vtk', 'u_pressure_opt_with_JAX_body_before_opt_deformed_bar_untangle.vtu')
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


volumes = [tet_signed_volume(*current_mesh.points[cell]) for cell in cells]

avg_volume = onp.mean(volumes)
min_volume = onp.min(volumes)
print("Average volume:", avg_volume)
print("Minimum volume:", min_volume)

for iteration in range(max_iterations):

    # params = np.zeros_like(original_cood)
    params = params * 0

    # Step 1: Compute gradient
    d_coord = jax.grad(composed_fn)(params)
    # What array is occupying the memory 
    grad_norm = np.linalg.norm(d_coord)
    if grad_norm > 1e-8:
        d_coord = d_coord / grad_norm

    params  = params - learning_rate * d_coord

    updated_mesh_points = onp.copy(current_mesh.points)
    updated_mesh_points[non_fixed_indices] = updated_mesh_points[non_fixed_indices] + params

    # current_mesh = current_mesh_dummy
    current_mesh.points = updated_mesh_points

    # del problem, fwd_pred, sol_list # Replace with variables no longer needed
    # gc.collect()

    problem.mesh[0].points = updated_mesh_points
        # Step 5: Solve the FEM problem with the updated geometry
    
    sol_list = fwd_pred(params)
    if sol_list == None:
        print('Negative J')
        print('Negative J')
        print('Negative J')
        print('Negative J')
        print('Negative J')
        print('Negative J')
        print('Negative J')
        print('Negative J')

        


        # params = params * 0
        # problem.mesh[0].points = updated_mesh_points
        # sol_list = fwd_pred(params)
        # u_sol_opt = sol_list[0]
        # vtk_path_opt = os.path.join(data_dir, 'vtk', 'u_pressure_opt_with_JAX_body_hemi_fine_just_befor_negative_J.vtu')
        # os.makedirs(os.path.dirname(vtk_path_opt), exist_ok=True)
        # save_sol(problem.fes[0], u_sol_opt, vtk_path_opt)
        # break
        updated_mesh_points = onp.copy(current_mesh.points)
        updated_mesh_points[non_fixed_indices] = updated_mesh_points[non_fixed_indices] - params
        
        onp_fixed_bc_mask = onp.copy(fixed_bc_mask)
        new_points, changes = global_untangle_tet_mesh(updated_mesh_points, cells, onp_fixed_bc_mask, vol_threshold=1e-6, scale_factor=1.05)

        problem.mesh[0].points = new_points

        params = params * 0
        sol_list = fwd_pred(params)
        print('sol_list')
        print(sol_list)
        while(sol_list == None):
            new_points, changes = global_untangle_tet_mesh(new_points, cells, onp_fixed_bc_mask, vol_threshold=1e-6, scale_factor=1.1)
            problem.mesh[0].points = new_points
            sol_list = fwd_pred(params)
            print('UnTangle Boost')
            print('UnTangle Boost')
            print('UnTangle Boost')
            print('UnTangle Boost')

        learning_rate = learning_rate * 0.5
        print('learning_rate')
        print(learning_rate)
        relax_flag = relax_flag + 1

        if relax_flag > 4:
            print('Relax Relax Relax Relax')
            print('Relax Relax Relax Relax')
            print('Relax Relax Relax Relax')
            print('Relax Relax Relax Relax')

    else:
        relax_flag = 0
        learning_rate = learning_rate * 1.05
        print('learning_rate')
        print(learning_rate)
    #####
        # u_sol_opt = sol_list[0]
        # vtk_path_opt = os.path.join(data_dir, 'vtk', 'u_pressure_opt_with_JAX_body_hemi_during_Iteration.vtu')
        # os.makedirs(os.path.dirname(vtk_path_opt), exist_ok=True)
        # save_sol(problem.fes[0], u_sol_opt, vtk_path_opt)
    #######
        # Step 6: Compute the cost for convergence
        current_cost = np.sum((((sol_list[0]+problem.mesh[0].points) - observed_positions_2))**2)
        cost_history.append([iteration, current_cost])  # Save iteration and cost
        print("//////////////////////////")
        print("//////////////////////////")
        print("//////////////////////////")
        print("//////////////////////////")
        print(f"Iteration {iteration}: Cost = {current_cost}")
        print("//////////////////////////")
        print("//////////////////////////")
        print("//////////////////////////")
        print("//////////////////////////")

        # if current_cost < 2:
        #     learning_rate = 0.01

            # Check for convergence
        # if iteration > 0 and np.abs(prev_cost - current_cost) < tolerance:
        if iteration > 0 and np.abs(current_cost) < tolerance:
            print("Converged!")
            break

        prev_cost = current_cost




     #memory_threshold = 70.0  # In percentage -- 19456

    # # Check system memory usage
    memory_info = psutil.virtual_memory()
    if memory_info.percent > memory_threshold:
        print(f"Memory usage exceeded {memory_threshold}%. Clearing caches...")
        jax.clear_caches()
        gc.collect()




# Save cost history to a CSV file
with open("cost_history_learning_rate_body_deformed_bar_untangle.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "Cost"])  # Write header
    writer.writerows(cost_history)         # Write data


end = time.time()
print('Run Time:')
print((end - start)/60)

u_sol_opt = sol_list[0]
vtk_path_opt = os.path.join(data_dir, 'vtk', 'u_pressure_opt_with_JAX_body_deformed_bar_untangle.vtu')
os.makedirs(os.path.dirname(vtk_path_opt), exist_ok=True)
save_sol(problem.fes[0], u_sol_opt, vtk_path_opt)

# Convert to pandas DataFrame
undeformed_df = pd.DataFrame(undeformed_coord, columns=["X", "Y", "Z"])
initial_gauss_df = pd.DataFrame(observed_positions_2, columns=["X", "Y", "Z"])
optimized_df = pd.DataFrame(current_mesh.points, columns=["X", "Y", "Z"])


# Save to CSV files
undeformed_df.to_csv("/home/gusdh/jax-fem/demos/hyperelasticity/jax-fem/demos/prac_hy/data/By_SciPy/undeformed_coordinates_JAX_body_deformed_bar_untangle.csv", index=False)
initial_gauss_df.to_csv("/home/gusdh/jax-fem/demos/hyperelasticity/jax-fem/demos/prac_hy/data/By_SciPy/initial_gauss_coordinates_body_deformed_bar_untangle.csv", index=False)
optimized_df.to_csv("/home/gusdh/jax-fem/demos/hyperelasticity/jax-fem/demos/prac_hy/data/By_SciPy/optimized_coordinates_body_deformed_bar_untangle.csv", index=False)

print('changes')
print(changes)