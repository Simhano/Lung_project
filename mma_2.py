import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh
from jax_fem.mma import optimize

# Define constitutive relationship.
class HyperElasticity(Problem):
    def __init__(self, *args, internal_pressure=2.0, **kwargs):
        self.internal_pressure = internal_pressure  # Make internal pressure variable
        super().__init__(*args, **kwargs)
        # No need to call self.custom_init(), it's called in __post_init__

    def custom_init(self):
        # Define additional attributes required by the optimizer.
        self.mesh = self.meshes[0]  # Access the first mesh
        self.points = self.mesh.points
        self.cells = self.mesh.cells
        self.flex_inds = np.arange(len(self.cells))
        self.non_fixed_nodes = self.additional_info[0]  # Get non-fixed nodes

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

        def first_PK_stress(u_grad, theta):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress

    def get_surface_maps(self):
        # Define the traction boundary conditions.
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

        return [x_0_traction, y_0_traction, y_1_traction, z_0_traction, z_1_traction]

    def set_params(self, params):
        # For this problem, params are the displacements of non-fixed nodes.
        # We need to reconstruct the full initial displacements.
        initial_displacements = self.mesh.points.copy()
        initial_displacements[self.non_fixed_nodes] = params.reshape((-1, 3))
        self.mesh.points = initial_displacements

    def compute_compliance(self, sol, observed_positions):
        # Compute the loss between the simulated positions and observed positions.
        simulated_positions = self.mesh.points + sol
        loss = np.sum((simulated_positions[self.non_fixed_nodes] - observed_positions[self.non_fixed_nodes]) ** 2)
        return loss

# Do some cleaning work. Remove old solution files.
data_dir = os.path.join(os.path.dirname(__file__), 'data')
files = glob.glob(os.path.join(data_dir, 'vtk/*'))
for f in files:
    os.remove(f)

# Specify mesh-related information.
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
Lx, Ly, Lz = 1., 1., 1.
meshio_mesh = box_mesh_gmsh(Nx=3, Ny=3, Nz=3, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir, ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# Define boundary conditions and values.
def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[right] * 3,
                     [0, 1, 2],
                     [zero_dirichlet_val] * 3]

# Define traction boundary conditions.
def traction_x_0(point):
    return np.isclose(point[0], 0.0, atol=1e-5)

def traction_y_0(point):
    return np.isclose(point[1], 0.0, atol=1e-5)

def traction_y_1(point):
    return np.isclose(point[1], Ly, atol=1e-5)

def traction_z_0(point):
    return np.isclose(point[2], 0.0, atol=1e-5)

def traction_z_1(point):
    return np.isclose(point[2], Lz, atol=1e-5)

location_fns = [traction_x_0, traction_y_0, traction_y_1, traction_z_0, traction_z_1]

# Generate observed data.

# Forward for pressure = 2
internal_pressure_2 = 2.0

problem_2 = HyperElasticity(mesh,
                            vec=3,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns=location_fns,
                            internal_pressure=internal_pressure_2)

# Solve the problem
sol_list_2 = solver(problem_2, solver_options={'petsc_solver': {}})

# Store the solution
u_sol_2 = sol_list_2[0]
vtk_path_2 = os.path.join(data_dir, 'vtk', 'u_pressure_2.vtu')
os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
save_sol(problem_2.fes[0], u_sol_2, vtk_path_2)

# Forward for pressure = 3
internal_pressure_3 = 3.0

problem_3 = HyperElasticity(mesh,
                            vec=3,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns=location_fns,
                            internal_pressure=internal_pressure_3)

# Solve the problem
sol_list_3 = solver(problem_3, solver_options={'petsc_solver': {}})

# Store the solution
u_sol_3 = sol_list_3[0]
vtk_path_3 = os.path.join(data_dir, 'vtk', 'u_pressure_3.vtu')
os.makedirs(os.path.dirname(vtk_path_3), exist_ok=True)
save_sol(problem_3.fes[0], u_sol_3, vtk_path_3)

# Inverse Problem Starts

# Data-set
original_positions = meshio_mesh.points

# Convert to standard NumPy array
mesh_points_onp = onp.array(meshio_mesh.points)

# Identify fixed nodes
fixed_nodes = onp.isclose(mesh_points_onp[:, 0], Lx, atol=1e-5)
non_fixed_nodes = np.where(~fixed_nodes)[0]

# Data-set for optimization
observed_positions_2 = mesh_points_onp + onp.array(u_sol_2)
observed_positions_3 = mesh_points_onp + onp.array(u_sol_3)

# Define the optimization problem
class OptimizationProblem:
    def __init__(self, mesh, observed_positions_2, observed_positions_3, non_fixed_nodes):
        self.mesh = mesh
        self.observed_positions_2 = observed_positions_2
        self.observed_positions_3 = observed_positions_3
        self.non_fixed_nodes = non_fixed_nodes
        self.xval = mesh.points[non_fixed_nodes].flatten()  # Initial guess (positions of non-fixed nodes)
        self.nelx = len(self.xval)
        self.dim = 3  # 3D problem

    def objective(self, x):
        # Reconstruct the full initial positions
        initial_displacements = self.mesh.points.copy()
        initial_displacements[self.non_fixed_nodes] = x.reshape((-1, self.dim))

        # Create a mesh with updated positions
        current_mesh = Mesh(initial_displacements, self.mesh.cells)

        # Solve forward problems with pressures 2 and 3
        problem_2 = HyperElasticity(current_mesh,
                                    vec=3,
                                    dim=3,
                                    ele_type=ele_type,
                                    dirichlet_bc_info=dirichlet_bc_info,
                                    location_fns=location_fns,
                                    internal_pressure=2.0,
                                    additional_info=[self.non_fixed_nodes])
        fwd_pred_2 = ad_wrapper(problem_2, solver_options={'petsc_solver': {}}, adjoint_solver_options={'petsc_solver': {}})
        sol_list_2 = fwd_pred_2()
        u_sol_2 = sol_list_2[0]

        problem_3 = HyperElasticity(current_mesh,
                                    vec=3,
                                    dim=3,
                                    ele_type=ele_type,
                                    dirichlet_bc_info=dirichlet_bc_info,
                                    location_fns=location_fns,
                                    internal_pressure=3.0,
                                    additional_info=[self.non_fixed_nodes])
        fwd_pred_3 = ad_wrapper(problem_3, solver_options={'petsc_solver': {}}, adjoint_solver_options={'petsc_solver': {}})
        sol_list_3 = fwd_pred_3()
        u_sol_3 = sol_list_3[0]

        # Calculate simulated positions
        simulated_positions_2 = initial_displacements + u_sol_2
        simulated_positions_3 = initial_displacements + u_sol_3

        # Compute the loss using only non-fixed nodes
        loss_2 = np.sum((simulated_positions_2[self.non_fixed_nodes] - self.observed_positions_2[self.non_fixed_nodes]) ** 2)
        loss_3 = np.sum((simulated_positions_3[self.non_fixed_nodes] - self.observed_positions_3[self.non_fixed_nodes]) ** 2)

        regularization = np.sum((initial_displacements - self.mesh.points)**2)

        total_loss = loss_2 + loss_3 + 0.1 * regularization
        print("Total Loss:", total_loss)
        return total_loss

    def objective_grad(self, x):
        J, gradJ = jax.value_and_grad(self.objective)(x)
        return J, gradJ

    def constraints(self, x, epoch):
        # No constraints in this problem
        c = np.array([])
        gradc = np.array([[]])
        return c, gradc

# Initialize the optimization problem
opt_problem = OptimizationProblem(mesh, observed_positions_2, observed_positions_3, non_fixed_nodes)

# Prepare the initial design variables
x_ini = opt_problem.xval.copy().reshape((-1, 1))

# Define the number of constraints
numConstraints = 0

# Set optimization parameters
optimizationParams = {'maxIters': 50, 'movelimit': 0.1}

# Define the objective and constraint handles
def objectiveHandle(x):
    J, dJ = opt_problem.objective_grad(x)
    return J, dJ

def consHandle(x, epoch):
    c, gradc = opt_problem.constraints(x, epoch)
    return c, gradc

# Run the MMA Optimization
optimize(opt_problem, x_ini, optimizationParams, objectiveHandle, consHandle, numConstraints)