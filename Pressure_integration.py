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
import numpy as onp

# Define constitutive relationship.
class HyperElasticity(Problem):
    def __init__(self, mesh, pressure=2, **kwargs):
        self.p0 = float(pressure)
        super().__init__(mesh, **kwargs)

        # --- compute per-face, per-quad unit normals once ---

        # assume b_inds = boundary_inds_list[0]  # shape (n_faces, 2): (cell_index, face_index)
        # inside your HyperElasticity __init__ or custom_init:
        b_inds   = self.boundary_inds_list[0]   # shape (n_faces, 2)
        cell_ids = b_inds[:, 0]                    # which global cell each face lives on
        face_ids = b_inds[:, 1]                    # which local face (0…5) of that cell

        # pull out the 8 node‐IDs for each cell
        cells8   = self.mesh[0].cells[cell_ids]  # (n_faces, 8)


        hex8_face_nodes = self.fes[0].face_inds


        # pick exactly the 4 node‐IDs for each boundary face
        local_nodes   = hex8_face_nodes[face_ids]                  # (n_faces,4)
        cells8        = self.mesh[0].cells[cell_ids]

        # face_node_ids = cells8[np.arange(len(face_ids))[:,None], local_nodes]  # (n_faces,4)
        face_node_ids = np.take_along_axis(cells8, local_nodes, axis=1)
        # now grab their coordinates
        coords = self.mesh[0].points[face_node_ids]  # (n_faces,4,3)

        # form two edge vectors per face and cross to get area‐weighted normals
        v1 = coords[:,2,:] - coords[:,0,:]   # (n_faces,3)
        v2 = coords[:,1,:] - coords[:,0,:]   # (n_faces,3)
        # n0 = np.cross(v1, v2)                # (n_faces,3)
        n0 = np.cross(v1, v2)                # (n_faces,3)
        # jax.debug.print("coords[:,0,:]: {}", coords[:,0,:])
        # jax.debug.print("coords[:,1,:]: {}", coords[:,1,:])
        # jax.debug.print("coords[:,2,:]: {}", coords[:,2,:])
        # jax.debug.print("coords[:,3,:]: {}", coords[:,3,:])
        # normalize
        unit_n0 = n0 / np.linalg.norm(n0, axis=1, keepdims=True)  # (n_faces,3)


        ######### Normal vector dirction decision
        #    cells8.shape == (n_faces, n_nodes_per_cell, 3)
        cell_pts      = self.mesh[0].points[cells8]       # (n_faces, 8, 3) for hexes (or (n_faces,4,3) for tets)
        cell_centroid = np.mean(cell_pts, axis=1)         # (n_faces, 3)

        # 2) get the face‐centroids
        #    coords.shape == (n_faces, 4, 3)  (or (n_faces,3,3) for a triangular face)
        face_centroid = np.mean(coords, axis=1)           # (n_faces, 3)

        # 3) flip any normal that points “into” the cell
        to_face        = face_centroid - cell_centroid    # (n_faces, 3)
        jax.debug.print("to_face: {}", to_face)
        jax.debug.print("unit_n0: {}", unit_n0)
        dotp           = np.sum(to_face * unit_n0, axis=1, keepdims=True)  # (n_faces,1)
        signs          = np.sign(dotp)                    # +1 if already outward, –1 if inward
        unit_n0        = unit_n0 * signs                  # flip inward normals
        #########


        # finally, broadcast out to all quad points per face
        Q = self.physical_surface_quad_points[0].shape[1]
        unit_n0 = unit_n0[:, None, :].repeat(Q, axis=1)           # (n_faces, Q, 3)

        jax.debug.print("unit_n0: {}", unit_n0)
        # register for your surface map
        self.internal_vars_surfaces = [unit_n0]

        # jax.debug.print("unit_n0: {}", unit_n0)
        # selected_face_shape_grads_1 = []
        # for boundary_inds in self.boundary_inds_list:
        #     s_shape_grads_1 = []

        #     for fe in self.fes:
        #         # (num_selected_faces, num_face_quads, num_nodes, dim), (num_selected_faces, num_face_quads)
        #         face_shape_grads_physical_1, _ = fe.get_face_shape_grads(boundary_inds)  
        #         s_shape_grads_1.append(face_shape_grads_physical_1)
        #     # (num_selected_faces, num_face_quads, num_nodes + ..., dim)
        #     s_shape_grads_1 = np.concatenate(s_shape_grads_1, axis=2)

        #     selected_face_shape_grads_1.append(s_shape_grads_1)



        # # b_inds = self.boundary_inds_list[0]
        # face_shape_grads = selected_face_shape_grads_1[0]
        # t1 = face_shape_grads[:,:, 0, :]   # (n_faces, n_quads, dim)
        # t2 = face_shape_grads[:,:, 1, :]
        # n0 = np.cross(t1, t2)      # (n_faces, n_quads, dim)
        # unit_n0 = n0 / np.linalg.norm(n0, axis=-1, keepdims=True)

        # # length of this list must equal number of Neumann regions:
        # self.internal_vars_surfaces = [unit_n0]






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
        p0 = self.p0
        # signature: (u_quad, x_quad, normals_quad) -> traction_quad
        def surface_map(u, x, normals):
            
            # jax.debug.print("normals: {}", normals)
            normal = np.array([0.0, -1.0, 0.0])
            return -p0 * normals
            
        return [surface_map]


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

def pressure_face(point):
    x, y, z = point  # unpack the coordinates

    # use np.isclose for each face test
    on_left   = np.isclose(x, 0.0, atol=1e-5)
    # on_y_low  = np.isclose(y, 0.0, atol=1e-5)
    # on_y_high = np.isclose(y, Ly,  atol=1e-5)
    # on_z_low  = np.isclose(z, 0.0, atol=1e-5)
    # on_z_high = np.isclose(z, Lz,  atol=1e-5)

    # combine with element‐wise OR
    return on_left #| on_y_low | on_y_high | on_z_low | on_z_high


# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.


def dirichlet_val_x2(point):
    return (0.5 + (point[1] - 0.5) * np.cos(np.pi / 3.) -
            (point[2] - 0.5) * np.sin(np.pi / 3.) - point[1]) / 2.


def dirichlet_val_x3(point):
    return (0.5 + (point[1] - 0.5) * np.sin(np.pi / 3.) +
            (point[2] - 0.5) * np.cos(np.pi / 3.) - point[2]) / 2.


mask = ((mesh.points[:,0] < 0.01) 
        | (mesh.points[:,1] < 0.01)  
        | (mesh.points[:,2] < 0.01)
        | (mesh.points[:,1] > 0.9)
        | (mesh.points[:,2] > 0.9))

y_0_point = np.where(mask)[0]
    
def y_0(point, ind):
    return np.isin(ind, y_0_point)
 
# dirichlet_bc_info = [[left] * 3 + [right] * 3, # Results in [left, left, left] + [right, right, right] 
#                      [0, 1, 2] + [0, 1, 2], # Components: [u_x, u_y, u_z] 
#                      [zero_dirichlet_val, dirichlet_val_x2, dirichlet_val_x3] + [zero_dirichlet_val] * 3]

dirichlet_bc_info = [[right] * 3 , # Results in [left, left, left] + [right, right, right] 
                     [0, 1, 2] , # Components: [u_x, u_y, u_z] 
                     [zero_dirichlet_val] * 3 ]

location_fns = [y_0]




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


# print(meshio_mesh.cells_dict[cell_type])
# print(sol_list[0])