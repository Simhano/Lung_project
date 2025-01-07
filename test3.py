import jax
import jax.numpy as np
import os
from jax import Array
from jax import jit
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from jaxopt import ScipyMinimize
import numpy as onp

# Define constitutive relationship
class HyperElasticity(Problem):
    def __init__(self, *args, internal_pressure=2.0, **kwargs):
        self.internal_pressure = internal_pressure  # Internal pressure variable
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
        internal_pressure = self.internal_pressure

        def traction_fn(normal):
            def compute_traction(u, x):
                return -internal_pressure * normal
            return compute_traction

        # Define normal vectors for each boundary
        normals = [
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0])
        ]

        return [traction_fn(normal) for normal in normals]

# Mesh setup
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh_gmsh(Nx=3, Ny=3, Nz=3, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Boundary locations
traction_x_0_point = np.where(mesh.points[:, 0] == 0)[0]
traction_y_0_point = np.where(mesh.points[:, 1] == 0)[0]
traction_y_1_point = np.where(mesh.points[:, 1] == Ly)[0]
traction_z_0_point = np.where(mesh.points[:, 2] == 0)[0]
traction_z_1_point = np.where(mesh.points[:, 2] == Lz)[0]

# Define boundary locations
@jit
def boundary_condition(point, ind, traction_points):
    return np.isin(ind, traction_points)

# Define Dirichlet boundary values
def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[lambda point: np.isclose(point[0], Lx, atol=1e-5)] * 3,
                     [0, 1, 2],
                     [zero_dirichlet_val] * 3]

location_fns = [
    lambda point, ind: boundary_condition(point, ind, traction_x_0_point),
    lambda point, ind: boundary_condition(point, ind, traction_y_0_point),
    lambda point, ind: boundary_condition(point, ind, traction_y_1_point),
    lambda point, ind: boundary_condition(point, ind, traction_z_0_point),
    lambda point, ind: boundary_condition(point, ind, traction_z_1_point)
]

# Solve the forward problems for two pressures
# @jit
def solve_problem_for_pressure(mesh, pressure):
    problem = HyperElasticity(
        mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
        internal_pressure=pressure
    )
    sol_list = solver(problem, solver_options={'petsc_solver': {}})
    return sol_list[0]

# Pressure 2
internal_pressure_2 = 2.0
u_sol_2 = solve_problem_for_pressure(mesh, internal_pressure_2)
vtk_path_2 = os.path.join(data_dir, 'vtk', 'u_pressure_2.vtu')
save_sol(mesh, u_sol_2, vtk_path_2)

# Pressure 3
internal_pressure_3 = 3.0
u_sol_3 = solve_problem_for_pressure(mesh, internal_pressure_3)
vtk_path_3 = os.path.join(data_dir, 'vtk', 'u_pressure_3.vtu')
save_sol(mesh, u_sol_3, vtk_path_3)

# Observed and original positions
observed_positions_2 = meshio_mesh.points + onp.array(u_sol_2)
observed_positions_3 = meshio_mesh.points + onp.array(u_sol_3)
original_positions = np.array(meshio_mesh.points)

# Identify fixed and non-fixed nodes
fixed_nodes = onp.isclose(meshio_mesh.points[:, 0], Lx, atol=1e-5)
non_fixed_nodes = ~fixed_nodes

# Loss function for optimization
# @jit
def loss_function(init_non_fixed_nodes_flat):
    original_nodes = original_positions.at[non_fixed_nodes].set(init_non_fixed_nodes_flat.reshape((-1, 3)))

    # Forward problem for pressure = 2
    simulated_positions_2 = original_nodes + solve_problem_for_pressure(Mesh(original_nodes, mesh.cells), internal_pressure_2)

    # Forward problem for pressure = 3
    simulated_positions_3 = original_nodes + solve_problem_for_pressure(Mesh(original_nodes, mesh.cells), internal_pressure_3)

    # Compute the loss
    loss_2 = np.sum((simulated_positions_2[non_fixed_nodes] - observed_positions_2[non_fixed_nodes])**2)
    loss_3 = np.sum((simulated_positions_3[non_fixed_nodes] - observed_positions_3[non_fixed_nodes])**2)

    return loss_2 + loss_3

# Initial guess
init_non_fixed_nodes_flat = observed_positions_2[non_fixed_nodes].flatten()

# Define the objective function for optimization
def objective_function(x):
    return loss_function(x)

# Optimize using ScipyMinimize
optimizer = ScipyMinimize(fun=objective_function)
opt_results = optimizer.run(init_params=init_non_fixed_nodes_flat)
optimized_original_nodes_flat = opt_results.params
optimized_original_nodes = optimized_original_nodes_flat.reshape(meshio_mesh.points.shape)

# Display the results
print("Optimization complete.")
print("Optimized original node positions:")
print(optimized_original_nodes)
