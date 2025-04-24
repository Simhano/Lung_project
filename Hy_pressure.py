# Import some useful modules.
import jax
import jax.numpy as np
import os


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
        internal_pressure = 2  # N/m²   (t = -pn)

        def x_left_traction(u, x):
            normal = np.array([-1.0, 0.0, 0.0])
            return -internal_pressure * normal
        def y_down_traction(u, x):
            normal = np.array([0., -1.0, 0.0])
            return -internal_pressure * normal
        def y_up_traction(u, x):
            normal = np.array([0.0, 1.0, 0.0])
            return -internal_pressure * normal
        def z_down_traction(u, x):
            normal = np.array([0.0, 0.0, -1.0])
            return -internal_pressure * normal
        def z_up_traction(u, x):
            normal = np.array([0.0, 0.0, 1.0])
            jax.debug.print("normal: {}", normal)
            return -internal_pressure * normal
        return [x_left_traction, y_down_traction, y_up_traction, z_down_traction, z_up_traction]


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

# Create an instance of the problem.
problem = HyperElasticity(mesh,
                          vec=3,
                          dim=3,
                          ele_type=ele_type,
                          dirichlet_bc_info=dirichlet_bc_info,
                          location_fns=location_fns)
# Solve the defined problem.    
sol_list,_ = solver(problem, solver_options={'petsc_solver': {}})

# Store the solution to local file.
vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
save_sol(problem.fes[0], sol_list[0], vtk_path)



# (num_cells, num_quads, vec, dim)
# u_grad = problem.fes[0].sol_to_grad(sol_list[0])
# print(u_grad+np.eye(problem.dim))
# print(u_grad.shape)


print(meshio_mesh.cells_dict[cell_type])
print(meshio_mesh.points)
