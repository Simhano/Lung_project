# Import some useful modules.
import jax
import jax.numpy as np
import os
import meshio

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh


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

    # Define the source term b
    def get_mass_map(self):
        def mass_map(u, x):
            density = 1000  # kg/m³, breast tissue density
            g = 9.81  # m/s², gravitational acceleration
            val = np.array([0.0, 0.0, 0.2])
            return val
        return mass_map

# Specify mesh-related information (first-order hexahedron element).
# ele_type = 'TET10'
ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 4.5, 3., 3.
meshio_mesh = box_mesh_gmsh(Nx=3,
                       Ny=3,
                       Nz=3,
                       Lx=Lx,
                       Ly=Ly,
                       Lz=Lz,
                       data_dir=data_dir,
                       ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# meshio_mesh = meshio.read("hemi_tet.inp")
# mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
# mesh.ele_type = 'hexahedron'
print(mesh.points)
print(mesh.cells)
# print(mesh.cell_type)
print(mesh.points)

# Define boundary locations.

# def right(point):
#     # print(np.isclose(point[0], Lx, atol=1e-5))
#     return np.isclose(point[0], Lx, atol=1e-5)

def left(point):
    # print(np.isclose(point[0], Lx, atol=1e-5))
    return np.isclose(point[0], 0, atol=1e-5)

def y_0(point):
    # print(np.isclose(point[0], Lx, atol=1e-5))
    return np.isclose(point[1], 0, atol=1e-5)


# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[left] *3, # Results in [left, left, left] + [right, right, right] 
                     [0, 1, 2], # Components: [u_x, u_y, u_z] 
                     [zero_dirichlet_val] * 3]

# location_fns = [traction_x_0, traction_y_0, traction_y_1, traction_z_0, traction_z_1]


# Create an instance of the problem.
problem = HyperElasticity(mesh,
                          vec=3,
                          dim=3,
                          ele_type=ele_type,
                          dirichlet_bc_info=dirichlet_bc_info)
# Solve the defined problem.    
sol_list = solver(problem, solver_options={'petsc_solver': {}})

# Store the solution to local file.
vtk_path = os.path.join(data_dir, f'vtk/u_body_forwawrd_befor_inter_for_comp.vtu')
save_sol(problem.fes[0], sol_list[0], vtk_path)

print("mesh.points")
print(mesh.points)

print("sol_list[0]")
print(sol_list[0])