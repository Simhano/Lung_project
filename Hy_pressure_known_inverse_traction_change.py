# Import some useful modules.
import jax
import jax.numpy as np
import os
import numpy as onp

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
from jax import value_and_grad
import scipy.optimize


# Define constitutive relationship.
class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.

    def __init__(self, *args, internal_pressure=None, pressure_node_indices=None,
                 left_face_nodes=None, y_down_nodes=None, y_up_nodes=None,
                 z_down_nodes=None, z_up_nodes=None, **kwargs):
        
        self.internal_pressure = internal_pressure  # Assign internal pressure
        self.pressure_node_indices = pressure_node_indices
        self.left_face_nodes = left_face_nodes
        self.y_down_nodes = y_down_nodes
        self.y_up_nodes = y_up_nodes
        self.z_down_nodes = z_down_nodes
        self.z_up_nodes = z_up_nodes
        super().__init__(*args, **kwargs)

        # Debug statements
        print(f"self.mesh type: {type(self.mesh)}")
        if hasattr(self.mesh, 'points'):
            print(f"self.mesh.points shape: {self.mesh.points.shape}")
        else:
            print("self.mesh does not have a 'points' attribute.")


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

        # Ensure node indices are provided
        if self.pressure_node_indices is None:
            raise ValueError("Pressure node indices must be provided.")

        # Get normals
        # normals = self.get_normals()

        # Prepare list of surface maps
        surface_maps = []
        
        # Create a list of tuples (node_indices, normal) for pressure-applied surfaces
        surfaces = [
            (self.left_face_nodes, np.array([-1.0, 0.0, 0.0])),
            (self.y_down_nodes, np.array([0.0, -1.0, 0.0])),
            (self.y_up_nodes, np.array([0.0, 1.0, 0.0])),
            (self.z_down_nodes, np.array([0.0, 0.0, -1.0])),
            (self.z_up_nodes, np.array([0.0, 0.0, 1.0]))
        ]

        for face_nodes, normal in surfaces:
            if len(face_nodes) > 0:
                def pressure_traction(u, x, face_nodes=face_nodes, normal=normal):
                    # """
                    # Applies traction to specified nodes.

                    # Parameters:
                    # - u (array): Displacement vector.
                    # - x (array): Position vector.
                    # - face_nodes (array-like): Indices of nodes on the current face.
                    # - normal (array-like): Normal vector of the current face.
                    
                    # Returns:
                    # - tractions (array): Traction vector applied to nodes.
                    # """
                    # Initialize traction as zero for all nodes
                    tractions = np.zeros_like(u)
                    # Apply pressure to the specified nodes
                    tractions = tractions.at[face_nodes].set(-internal_pressure * normal)
                    return tractions
                surface_maps.append(pressure_traction)

        return surface_maps

    def get_normals(self):
        # """
        # Defines normals for nodes based on initial configuration.

        # Returns:
        # - normals (array): Normal vectors for each node.
        # """
        # Initialize normals as zero
        normals = np.zeros_like(self.mesh.points)  # shape (num_nodes, dim)

        # Assign normals to nodes on each face
        normals = normals.at[self.left_face_nodes].set(np.array([-1.0, 0.0, 0.0]))
        normals = normals.at[self.y_down_nodes].set(np.array([0.0, -1.0, 0.0]))
        normals = normals.at[self.y_up_nodes].set(np.array([0.0, 1.0, 0.0]))
        normals = normals.at[self.z_down_nodes].set(np.array([0.0, 0.0, -1.0]))
        normals = normals.at[self.z_up_nodes].set(np.array([0.0, 0.0, 1.0]))

        return normals
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

print(mesh.points)
# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def y_down(point):
    return np.isclose(point[1], 0, atol=1e-5)

def y_up(point):
    return np.isclose(point[1], Ly, atol=1e-5)

def z_down(point):
    return np.isclose(point[2], 0, atol=1e-5)

def z_up(point):
    return np.isclose(point[2], Lz, atol=1e-5)




# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.


def dirichlet_val_x2(point):
    return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
            (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.


def dirichlet_val_x3(point):
    return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
            (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.


# dirichlet_bc_info = [[left] * 3 + [right] * 3, # Results in [left, left, left] + [right, right, right] 
#                      [0, 1, 2] + [0, 1, 2], # Components: [u_x, u_y, u_z] 
#                      [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] + [zero_dirichlet_val] * 3]

dirichlet_bc_info = [[right] * 3, # Results in [left, left, left] + [right, right, right] 
                     [0, 1, 2], # Components: [u_x, u_y, u_z] 
                     [zero_dirichlet_val] * 3]

location_fns = [left, y_down, y_up, z_down, z_up]


# Initial nodal positions
initial_nodes = mesh.points.copy()
# Identify nodes on each face based on initial positions
left_face_nodes = onp.where(onp.isclose(initial_nodes[:, 0], 0.0, atol=1e-5))[0]
y_down_nodes = onp.where(onp.isclose(initial_nodes[:, 1], 0.0, atol=1e-5))[0]
y_up_nodes = onp.where(onp.isclose(initial_nodes[:, 1], Ly, atol=1e-5))[0]
z_down_nodes = onp.where(onp.isclose(initial_nodes[:, 2], 0.0, atol=1e-5))[0]
z_up_nodes = onp.where(onp.isclose(initial_nodes[:, 2], Lz, atol=1e-5))[0]

# Combine nodes for pressure boundary conditions
pressure_boundary_nodes = onp.unique(onp.concatenate([
    left_face_nodes, y_down_nodes, y_up_nodes, z_down_nodes, z_up_nodes
]))

# Nodes on the right face (x = Lx)
dirichlet_boundary_nodes = onp.where(onp.isclose(initial_nodes[:, 0], Lx, atol=1e-5))[0]
# -------------------------------------------------------------------------------------------------------------
# --------------------------------------------Generating Data-set ---------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------
# Forward for pressure = 2-------------------------------------------------------------------------------------
internal_pressure_2 = 2.0

problem_2 = HyperElasticity(
    mesh=mesh,
    vec=3,
    dim=3,
    ele_type=ele_type,
    dirichlet_bc_info=dirichlet_bc_info,
    internal_pressure=internal_pressure_2,  # Internal pressure
    pressure_node_indices=pressure_boundary_nodes,
    left_face_nodes=left_face_nodes,
    y_down_nodes=y_down_nodes,
    y_up_nodes=y_up_nodes,
    z_down_nodes=z_down_nodes,
    z_up_nodes=z_up_nodes
)

# Solve the problem
sol_list_2 = solver(problem_2, solver_options={'petsc_solver': {}})

# Store the solution
u_sol_2 = sol_list_2[0]
vtk_path_2 = os.path.join(data_dir, 'vtk', 'u_pressure_2.vtu')
os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
save_sol(problem_2.fes[0], u_sol_2, vtk_path_2)


# print(u_sol_2)

# -------------------------------------------------------------------------------------------------------------
# Forward for pressure = 3-------------------------------------------------------------------------------------

internal_pressure_3 = 3.0

problem_3 = HyperElasticity(
    mesh=mesh,
    vec=3,
    dim=3,
    ele_type=ele_type,
    dirichlet_bc_info=dirichlet_bc_info,
    internal_pressure=internal_pressure_3,  # Internal pressure
    pressure_node_indices=pressure_boundary_nodes,
    left_face_nodes=left_face_nodes,
    y_down_nodes=y_down_nodes,
    y_up_nodes=y_up_nodes,
    z_down_nodes=z_down_nodes,
    z_up_nodes=z_up_nodes
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
observed_positions_2 = meshio_mesh.points + u_sol_2
observed_positions_3 = meshio_mesh.points + u_sol_3
original_positions   = meshio_mesh.points


# -------------------------------------------------------------------------------------------------------------
# Lose Function -----------------------------------------------------------------------------------------------

def loss_function(optimize_nodes_flat):
    # Reshape the flat array back to node positions
    # Reconstruct full node positions
    original_nodes = np.zeros_like(mesh.points)

    # Set positions of fixed nodes (Dirichlet boundary)
    original_nodes = original_nodes.at[dirichlet_boundary_nodes].set(mesh.points[dirichlet_boundary_nodes])

    # Set positions of nodes being optimized
    original_nodes = original_nodes.at[optimize_node_indices].set(
        optimize_nodes_flat.reshape(-1, 3)
    )
    # Create a mesh with the current estimate of the original positions
    current_mesh = Mesh(original_nodes, mesh.cells)
    
    # Set up and solve the forward problem for pressure = 2
    problem_2 = HyperElasticity(
        mesh=current_mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        internal_pressure=internal_pressure_2,
        pressure_node_indices=pressure_boundary_nodes,
        left_face_nodes=left_face_nodes,
        y_down_nodes=y_down_nodes,
        y_up_nodes=y_up_nodes,
        z_down_nodes=z_down_nodes,
        z_up_nodes=z_up_nodes
    )
    # sol_list_2 = jax.lax.stop_gradient(solver(problem_2, solver_options={'petsc_solver': {}}))
    sol_list_2 = solver(problem_2, solver_options={'petsc_solver': {}})
    u_sol_2 = sol_list_2[0]
    simulated_positions_2 = original_nodes + u_sol_2
    
    # Set up and solve the forward problem for pressure = 3
    problem_3 = HyperElasticity(
        mesh=current_mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        internal_pressure=internal_pressure_3,
        pressure_node_indices=pressure_boundary_nodes,
        left_face_nodes=left_face_nodes,
        y_down_nodes=y_down_nodes,
        y_up_nodes=y_up_nodes,
        z_down_nodes=z_down_nodes,
        z_up_nodes=z_up_nodes
    )
    # sol_list_3 = jax.lax.stop_gradient(solver(problem_3, solver_options={'petsc_solver': {}}))
    sol_list_3 = solver(problem_3, solver_options={'petsc_solver': {}})
    u_sol_3 = sol_list_3[0]
    simulated_positions_3 = original_nodes + u_sol_3
    
    # Compute the loss as the sum of squared differences
    loss_2 = np.sum((simulated_positions_2 - observed_positions_2)**2)
    loss_3 = np.sum((simulated_positions_3 - observed_positions_3)**2)
    total_loss = loss_2 + loss_3

    print("total_loss: ")
    print(total_loss)

    return total_loss


# loss_and_grad = value_and_grad(loss_function)


# -------------------------------------------------------------------------------------------------------------
# Initial Guess -----------------------------------------------------------------------------------------------

# initial_nodes_flat = observed_positions_3.flatten()
# initial_nodes_flat = original_positions.flatten()

# Indices of nodes to optimize (exclude Dirichlet boundary nodes)
optimize_node_indices = onp.setdiff1d(onp.arange(mesh.points.shape[0]), dirichlet_boundary_nodes)

# Initial guess for optimization
# initial_nodes_flat = observed_positions_2[optimize_node_indices].flatten()

def optimize_original_nodes():
    # """
    # Optimizes the original node positions to minimize the loss function.

    # Returns:
    # - estimated_original_nodes (array): The optimized node positions.
    # """
    # Initial guess for optimization: positions of nodes being optimized from observed_positions_2
    initial_nodes_flat = observed_positions_2[optimize_node_indices].flatten()

    # Perform optimization using finite differences for gradient approximation
    result = scipy.optimize.minimize(
        fun=loss_function,             # Function to minimize
        x0=initial_nodes_flat,         # Initial guess
        method='L-BFGS-B',             # Optimization method
        jac='2-point',                 # Use finite differences to approximate gradient
        options={'disp': True, 'maxiter': 100}
    )

    # Reconstruct full node positions from optimized result
    estimated_original_nodes = jnp.zeros_like(mesh.points)
    # Set fixed nodes (Dirichlet boundary)
    estimated_original_nodes = estimated_original_nodes.at[dirichlet_boundary_nodes].set(
        mesh.points[dirichlet_boundary_nodes]
    )
    # Set optimized nodes
    estimated_original_nodes = estimated_original_nodes.at[optimize_node_indices].set(
        result.x.reshape(-1, 3)
    )
    return estimated_original_nodes
# Run the optimization
estimated_original_nodes = optimize_original_nodes()


print(estimated_original_nodes)
