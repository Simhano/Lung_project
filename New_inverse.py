# Import some useful modules.
import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt


# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver, ad_wrapper
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, box_mesh_gmsh


# Define constitutive relationship.
class HyperElasticity_opt(Problem):
    def __init__(self, *args,internal_pressure=2.0, **kwargs):
        self.internal_pressure = internal_pressure  # Make internal pressure variable
        super().__init__(*args, **kwargs)

    def custom_init(self):
        self.fe = self.fes[0]

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
        self.mesh[0].points = params[0]





class HyperElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM 
    # solves -div(f(u_grad)) = b. Here, we define f(u_grad) = P. Notice how we first 
    # define 'psi' (representing W), and then use automatic differentiation (jax.grad) 
    # to obtain the 'P_fn' function.

    def __init__(self, *args, internal_pressure=2.0, **kwargs):
        self.internal_pressure = internal_pressure  # Make internal pressure variable
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
        
        # def x_left_traction(u, x):
        #     normal = np.array([-1.0, 0, 0])
        #     return -internal_pressure * normal
        
        return [x_0_traction, y_0_traction, y_1_traction, z_0_traction, z_1_traction]



# Specify mesh-related information (first-order hexahedron element).

ele_type = 'HEX8'
cell_type = get_meshio_cell_type(ele_type)
data_dir = os.path.join(os.path.dirname(__file__), 'data')
Lx, Ly, Lz = 100., 100., 100.
meshio_mesh = box_mesh_gmsh(Nx=2,
                            Ny=2,
                            Nz=2,
                            Lx=Lx,
                            Ly=Ly,
                            Lz=Lz,
                            data_dir=data_dir,
                            ele_type=ele_type)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

# print(mesh.points)


# Define Dirichlet boundary values.
traction_x_0_point = np.where(mesh.points[:,0]==0)[0]
traction_y_0_point = np.where(mesh.points[:,1]==0)[0]
traction_y_1_point = np.where(mesh.points[:,1]==Ly)[0]
traction_z_0_point = np.where(mesh.points[:,2]==0)[0]
traction_z_1_point = np.where(mesh.points[:,2]==Lz)[0]

# Define boundary locations for traction.
def traction_x_0(point, ind):
    # print(np.isin(ind, traction_x_0_point) )
    return np.isin(ind, traction_x_0_point) 
def traction_y_0(point, ind):
    # print(ind, ind_set_traction)
    return np.isin(ind, traction_y_0_point) 
def traction_y_1(point, ind):
    # print(ind, ind_set_traction)
    return np.isin(ind, traction_y_1_point) 
def traction_z_0(point, ind):
    # print(ind, ind_set_traction)
    return np.isin(ind, traction_z_0_point) 
def traction_z_1(point, ind):
    # print(ind, ind_set_traction)
    return np.isin(ind, traction_z_1_point) 


# Define boundary locations.

def right(point):
    # print(np.isclose(point[0], Lx, atol=1e-5))
    return np.isclose(point[0], Lx, atol=1e-5)

# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.

dirichlet_bc_info = [[right] + [right] + [right], # Results in [left, left, left] + [right, right, right] 
                     [0, 1, 2], # Components: [u_x, u_y, u_z] 
                     [zero_dirichlet_val] + [zero_dirichlet_val] + [zero_dirichlet_val]]

location_fns = [traction_x_0, traction_y_0, traction_y_1, traction_z_0, traction_z_1]

internal_pressure_2 = 2.0

problem_2 = HyperElasticity(mesh,
                            vec=3,
                            dim=3,
                            ele_type=ele_type,
                            dirichlet_bc_info=dirichlet_bc_info,
                            location_fns=location_fns,
                            internal_pressure=internal_pressure_2
)

sol_list_2 = solver(problem_2, solver_options={'petsc_solver': {}})

# Store the solution
u_sol_2 = sol_list_2[0]
print(u_sol_2)
# vtk_path_2 = os.path.join(data_dir, 'vtk', 'u_one_node.vtu')
# os.makedirs(os.path.dirname(vtk_path_2), exist_ok=True)
# save_sol(problem_2.fes[0], u_sol_2, vtk_path_2)

# Create an instance of the problem.



problem = HyperElasticity_opt(mesh, vec=3, dim=3, ele_type=ele_type, dirichlet_bc_info=dirichlet_bc_info,
                          location_fns=location_fns, internal_pressure=internal_pressure_2)

# mesh_points_modi = mesh.points 8.79406 -3.2669
# scale_d = 1.
# original_cood = np.copy(mesh.points) * 1.1

original_cood = mesh.points
internal_pressure = 2.0

params = [original_cood]

print("HAHA")
# Implicit differentiation wrapper
fwd_pred = ad_wrapper(problem) 
print("HOHO")
sol_list = fwd_pred(params)
print(sol_list[0])
# vtk_path = os.path.join(data_dir, f'vtk/u.vtu')
# save_sol(problem.fe, sol_list[0], vtk_path)

def test_fn(sol_list):
    # print('test fun')
    print(sol_list[0])
    return np.sum((sol_list[0] - u_sol_2)**2)

def composed_fn(params):
    # print()
    return test_fn(fwd_pred(params))



d_coord= jax.grad(composed_fn)(params)
print(d_coord[0].shape)
print(d_coord)

