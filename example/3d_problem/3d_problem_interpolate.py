# %% 
#script to interpolate a 3d numpy array of values over a mesh.
import dolfinx as dlx
from mpi4py import MPI
import numpy as np
import sys
import os
import dolfinx.fem.petsc
import h5py

import ufl
from matplotlib import pyplot as plt
from typing import Dict
from typing import Sequence, Dict

sys.path.append(os.environ.get("HIPPYLIBX_BASE_DIR", "../../"))
import hippylibX as hpx

class DiffusionApproximation:
    def __init__(self, D: float, u0: float):
        """
        Define the forward model for the diffusion approximation to radiative transfer equations

        D: diffusion coefficient 1/mu_eff with mu_eff = sqrt(3 mu_a (mu_a + mu_ps) ), where mu_a
           is the unknown absorption coefficient, and mu_ps is the reduced scattering coefficient

        u0: Incident fluence (Robin condition)

        ds: boundary integrator for Robin condition
        """
        self.D = D
        self.u0 = u0
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})
        self.ds = ufl.Measure("ds", metadata={"quadrature_degree": 4})

    def __call__(
        self, u: dlx.fem.Function, m: dlx.fem.Function, p: dlx.fem.Function
    ) -> ufl.form.Form:
        return (
            ufl.inner(self.D * ufl.grad(u), ufl.grad(p))
            * ufl.dx(metadata={"quadrature_degree": 4})
            + ufl.exp(m) * ufl.inner(u, p) * self.dx
        )

class PACTMisfitForm:
    def __init__(self, d: float, sigma2: float):
        self.sigma2 = sigma2
        self.d = d
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})

    def __call__(self, u: dlx.fem.Function, m: dlx.fem.Function) -> ufl.form.Form:
        return (
            0.5
            / self.sigma2
            * ufl.inner(u * ufl.exp(m) - self.d, u * ufl.exp(m) - self.d)
            * self.dx
        )


sep = "\n" + "#" * 80 + "\n"
comm = MPI.COMM_WORLD
rank = comm.rank
nproc = comm.size
mesh_filename = 'submesh_3d_problem.xdmf'

fid = dlx.io.XDMFFile(comm, mesh_filename, "r")
msh = fid.read_mesh(name="mesh")
Vh_phi = dlx.fem.functionspace(msh, ("Lagrange", 2))
Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))
Vh = [Vh_phi, Vh_m, Vh_phi]
ndofs = [
    Vh_phi.dofmap.index_map.size_global * Vh_phi.dofmap.index_map_bs,
    Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs,
]

hpx.master_print(comm, sep, "Set up the mesh and finite element spaces", sep)
hpx.master_print(comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs))
Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))
m_true = dlx.fem.Function(Vh_m)

with h5py.File('downsample_labels.h5', 'r') as f:
        origin = f['new_origin'][:]
        voxel_size = f['new_voxel_size'][:].flatten()[0]

with h5py.File('downsample_optical_properties.h5', 'r') as f:
        optical_arr = f['reduced_opt_array'][:]



def interpolate_function(x):    
    # clip is needed due to floating point precision:
    # nearly 0 values obtained on subtraction of two 'equal' values
    # for 0 index cell and last index cell fall on eithier side, giving '-1'
    # and 'length' value 
    i = np.clip(np.floor((x[0] - origin[0]) / voxel_size).astype(int), 0, optical_arr.shape[0] - 1)
    j = np.clip(np.floor((x[1] - origin[1]) / voxel_size).astype(int), 0, optical_arr.shape[1] - 1)
    k = np.clip(np.floor((x[2] - origin[2]) / voxel_size).astype(int), 0, optical_arr.shape[2] - 1)
    
    return optical_arr[i, j, k]

    # Interpolate values
m_true.interpolate(interpolate_function)

nproc = comm.size
with dlx.io.XDMFFile(
    msh.comm,
    "interpolated_optical_properties_np{0:d}.xdmf".format(nproc),
    "w",
) as file:
    file.write_mesh(msh)
    file.write_function(m_true,0)

    # dirichlet B.C.
    # def flat_surface(x):
    
    # def curved_surface(x):
        
    
    
    
# uD = dlx.fem.Function(Vh[hpx.STATE])
# uD.interpolate(lambda x: x[1])
# uD.x.scatter_forward()

# def top_bottom_boundary(x: Sequence[float]) -> Sequence[bool]:
#     return np.logical_or(np.isclose(x[1], 1), np.isclose(x[1], 0))

# fdim = msh.topology.dim - 1
# top_bottom_boundary_facets = dlx.mesh.locate_entities_boundary(
#     msh, fdim, top_bottom_boundary
# )
# dirichlet_dofs = dlx.fem.locate_dofs_topological(
#     Vh[hpx.STATE], fdim, top_bottom_boundary_facets
# )
# bc = dlx.fem.dirichletbc(uD, dirichlet_dofs)

# # bc0
# uD_0 = dlx.fem.Function(Vh[hpx.STATE])
# uD_0.interpolate(lambda x: 0.0 * x[0])
# uD_0.x.scatter_forward()
# bc0 = dlx.fem.dirichletbc(uD_0, dirichlet_dofs)



    # # # FORWARD MODEL
    # f = dlx.fem.Constant(msh, dlx.default_scalar_type(0.0))
    # u0 = 1.0
    # D = 1.0 / 24.0
    # pde_handler = DiffusionApproximation(D, u0)
    # pde = hpx.PDEVariationalProblem(Vh, pde_handler, [bc], [bc0], is_fwd_linear=True)
    # # GROUND TRUTH
    # u_true = pde.generate_state()


    # optical_arr = np.load('3d_problem/reduced_opt_array_factor_8_method_zoom.npy') #(46, 103, 98)
    # offset_disp = np.load('3d_problem/offset_disp_factor_8_method_zoom.npy')

    # Vh_m = dlx.fem.functionspace(msh, ("Lagrange", 1))
    # m_true = dlx.fem.Function(Vh_m)

    # original_center = np.array([-85,-85,-85])
    # voxel_sizes = np.array([0.125, 0.125, 0.125])

    # #revsied centers:
    # offset = np.zeros(3)
    # for i in range(3):
    #     offset[i] = original_center[i] + voxel_sizes[i]*offset_disp[i]

    # def interpolate_function(x):

    #     i = np.clip(np.floor((x[0] - offset[0]) / voxel_sizes[0]).astype(int), 0, optical_arr.shape[0] - 1)
    #     j = np.clip(np.floor((x[1] - offset[1]) / voxel_sizes[1]).astype(int), 0, optical_arr.shape[1] - 1)
    #     k = np.clip(np.floor((x[2] - offset[2]) / voxel_sizes[2]).astype(int), 0, optical_arr.shape[2] - 1)
        
    #     return optical_arr[i, j, k]

    # # Interpolate values
    # m_true.interpolate(interpolate_function)
