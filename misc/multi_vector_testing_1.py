# import sys
# import os
# import numpy as np
# from mpi4py import MPI
# import dolfinx as dlx
# import ufl
# import petsc4py
# import dolfinx.fem.petsc
# import copy

# sys.path.append(os.path.abspath("../"))
# import hippylibX as hpx
# sys.path.append(os.path.abspath("../example"))
# from hippylibX.modeling.reducedHessian import ReducedHessian

# class MultiVector:
#     def __init__(self,example_vec=None, nvec=None, mv = None):
#         if mv is None: #original
#             self.data = []
#             self.nvec = nvec

#             for i in range(self.nvec):
#                 self.data.append(example_vec.duplicate())
#         else: #copy
#             self.nvec = mv.nvec
#             self.data = copy.deepcopy(mv.data)

#     def __del__(self):
#         for d in self.data:
#             d.destroy()

#     def __getitem__(self,k):
#         return self.data[k]

#     def scale(self, alpha):
#         if(isinstance(alpha,float)):
#             for d in self.data:
#                 d.scale(alpha)
#         elif(isinstance(alpha,np.array)):
#             for i,d in enumerate(self.data):
#                 d.scale(alpha[i])

#     def dot(self, v) -> np.array:
#         if(isinstance(v,petsc4py.PETSc.Vec)):
#             return_values = []
#             for i in range(self.nvec):
#                 return_values.append(self[i].dot(v)  )
#             return np.array(return_values)

#         elif(isinstance(v,MultiVector)):
#             return_values = []
#             for i in range(self.nvec):
#                 return_row = []
#                 for j in range(v.nvec):
#                     return_row.append(self[i].dot(v[j]))
#                 return_values.append(return_row)
#             return np.array(return_values)

#     def reduce(self, alpha: np.array) -> petsc4py.PETSc.Vec:
#         return_vec = self[0].duplicate()
#         return_vec.scale(0.)
#         for i in range(self.nvec):
#             return_vec.axpy(alpha[i],self[i])
#         return return_vec

#     def Borthogonalize(self,B):
#         return self._mgs_stable(B)

#     def _mgs_stable(self, B : petsc4py.PETSc.Mat):
#         n = self.nvec
#         Bq = MultiVector(self[0], n)
#         r  = np.zeros((n,n), dtype = 'd')
#         reorth = np.zeros((n,), dtype = 'd')
#         eps = np.finfo(np.float64).eps

#         for k in np.arange(n):
#             B.mult(self[k], Bq[k])
#             t = np.sqrt( Bq[k].dot(self[k]))

#             nach = 1;    u = 0;
#             while nach:
#                 u += 1
#                 for i in np.arange(k):
#                     s = Bq[i].dot(self[k])
#                     r[i,k] += s
#                     self[k].axpy(-s, self[i])

#                 B.mult(self[k], Bq[k])
#                 tt = np.sqrt(Bq[k].dot(self[k]))
#                 if tt > t*10.*eps and tt < t/10.:
#                     nach = 1;    t = tt;
#                 else:
#                     nach = 0;
#                     if tt < 10.*eps*t:
#                         tt = 0.

#             reorth[k] = u
#             r[k,k] = tt
#             if np.abs(tt*eps) > 0.:
#                 tt = 1./tt
#             else:
#                 tt = 0.

#             self[k].scale(tt)
#             Bq[k].scale(tt)

#         return Bq, r

# def MatMvMult(A, x, y):
#     assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
#     if hasattr(A,'matMvMult'):
#         A.matMvMult(x,y)
#     else:
#         for i in range(x.nvec()):
#             A.mult(x[i], y[i])

# def MatMvTranspmult(A, x, y):
#     assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
#     assert hasattr(A,'transpmult'), "A does not have transpmult method implemented"
#     if hasattr(A,'matMvTranspmult'):
#         A.matMvTranspmult(x,y)
#     else:
#         for i in range(x.nvec()):
#             A.multTranspose(x[i], y[i])

# def MvDSmatMult(X, A, Y):
#     assert X.nvec() == A.shape[0], "X Number of vecs incompatible with number of rows in A"
#     assert Y.nvec() == A.shape[1], "Y Number of vecs incompatible with number of cols in A"
#     for j in range(Y.nvec()):
#         Y[j].zero()
#         X.reduce(Y[j], A[:,j].flatten())


# nx = 1
# ny = 1
# noise_variance = 1e-6

# comm = MPI.COMM_WORLD
# rank = comm.rank
# nproc = comm.size
# msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
# Vh = dlx.fem.FunctionSpace(msh, ("Lagrange", 1))


# #1. Mass matrix: - check if its symmetric:
# trial = ufl.TrialFunction(Vh)
# test = ufl.TestFunction(Vh)
# varfM = ufl.inner(trial, test) *  ufl.Measure("dx", metadata={"quadrature_degree": 4})
# M = dlx.fem.petsc.assemble_matrix(dlx.fem.form(varfM))
# M.assemble()

# def petsc2array(v):
#     s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
#     return s

# # print(petsc2array(M))
# # is symmetric - so can just work with M:
# sample_vec1 = dlx.fem.Function(Vh)
# sample_vec1 = sample_vec1.x
# hpx.parRandom.normal(1., sample_vec1)
# # print(sample_vec1.array[:])

# # sample_vec2 = dlx.fem.Function(Vh)
# # sample_vec2 = sample_vec2.x
# # hpx.parRandom.normal(1., sample_vec2)

# # print(sample_vec1.array[:])
# # print(sample_vec2.array[:])

# # print(sample_vec1[:],'\n',sample_vec2[:])

# sample_petsc_vec = dlx.la.create_petsc_vector_wrap(sample_vec1)

# Omega = MultiVector(sample_petsc_vec,3)

# hpx.parRandom.normal(1.,Omega)
# # for d in Omega:
# #     d.assemble()

# # for i in range(Omega.nvec):
# #     print(Omega[i].getArray())

# Q = MultiVector(mv = Omega)
# Bq, r = Omega.Borthogonalize(M)

# # print(r)
# print("Omega\n")
# for i in range(Omega.nvec):
#     print(Omega[i].getArray())

# # print("Mass matrix\n")
# # print(petsc2array(M))

# # print(petsc2array(M))

# # print("Omega\n")
# # for i in range(Omega.nvec):
# #     print(Omega[i].getArray())

# print("Q\n")
# for i in range(Q.nvec):
#     print(Q[i].getArray())


# mult = Omega.dot(Q)

# print('\n',mult)

# #     master_print(comm, sep, "Set up the mesh and finite element spaces", sep)
# #     master_print(comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs))

# #     # FORWARD MODEL
# #     u0 = 1.0
# #     D = 1.0 / 24.0
# #     pde_handler = DiffusionApproximation(D, u0)

# #     pde = hpx.PDEVariationalProblem(Vh, pde_handler, [], [], is_fwd_linear=True)
# #     # GROUND TRUTH
# #     m_true = dlx.fem.Function(Vh_m)
# #     m_true.interpolate(
# #         lambda x: np.log(0.01)
# #         + 3.0 * (((x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 2.0) * (x[1] - 2.0)) < 1.0)
# #     )
# #     m_true.x.scatter_forward()

# #     m_true = m_true.x
# #     u_true = pde.generate_state()

# #     x_true = [u_true, m_true, None]
# #     pde.solveFwd(u_true, x_true)
# #     xfun = [dlx.fem.Function(Vhi) for Vhi in Vh]
# #     # LIKELIHOOD
# #     hpx.updateFromVector(xfun[hpx.STATE], u_true)
# #     u_fun_true = xfun[hpx.STATE]
# #     hpx.updateFromVector(xfun[hpx.PARAMETER], m_true)
# #     m_fun_true = xfun[hpx.PARAMETER]

# #     d = dlx.fem.Function(Vh[hpx.STATE])
# #     expr = u_fun_true * ufl.exp(m_fun_true)
# #     hpx.projection(expr, d)
# #     hpx.parRandom.normal_perturb(np.sqrt(noise_variance), d.x)
# #     d.x.scatter_forward()
# #     misfit_form = PACTMisfitForm(d, noise_variance)
# #     misfit = hpx.NonGaussianContinuousMisfit(Vh, misfit_form)
# #     prior_mean = dlx.fem.Function(Vh_m)
# #     prior_mean.x.array[:] = np.log(0.01)
# #     prior_mean = prior_mean.x

# #     prior = hpx.BiLaplacianPrior(
# #         Vh_m, prior_param["gamma"], prior_param["delta"], mean=prior_mean
# #     )
# #     model = hpx.Model(pde, prior, misfit)

# #     noise = prior.generate_parameter("noise")
# #     m0 = prior.generate_parameter(0)
# #     hpx.parRandom.normal(1.0, noise)
# #     prior.sample(noise, m0)

# #     # # #######################################

# #     intial_guess_m = prior.generate_parameter(0)
# #     intial_guess_m.array[:] = prior_mean.array[:]
# #     x = [
# #         model.generate_vector(hpx.STATE),
# #         intial_guess_m,
# #         model.generate_vector(hpx.ADJOINT),
# #     ]
# #     if rank == 0:
# #         print(sep, "Find the MAP point", sep)

# #     parameters = hpx.ReducedSpaceNewtonCG_ParameterList()
# #     parameters["rel_tolerance"] = 1e-6
# #     parameters["abs_tolerance"] = 1e-9
# #     parameters["max_iter"] = 500
# #     parameters["cg_coarse_tolerance"] = 5e-1
# #     parameters["globalization"] = "LS"
# #     parameters["GN_iter"] = 20
# #     if rank != 0:
# #         parameters["print_level"] = -1

# #     solver = hpx.ReducedSpaceNewtonCG(model, parameters)

# #     x = solver.solve(x)

# #     if solver.converged:
# #         master_print(comm, "\nConverged in ", solver.it, " iterations.")
# #     else:
# #         master_print(comm, "\nNot Converged")
# #     master_print(
# #         comm, "Termination reason: ", solver.termination_reasons[solver.reason]
# #     )
# #     master_print(comm, "Final gradient norm: ", solver.final_grad_norm)
# #     master_print(comm, "Final cost: ", solver.final_cost)
# #     optimizer_results = {}
# #     if (
# #         solver.termination_reasons[solver.reason]
# #         == "Norm of the gradient less than tolerance"
# #     ):
# #         optimizer_results["optimizer"] = True
# #     else:
# #         optimizer_results["optimizer"] = False


# #     model.setPointForHessianEvaluations(x, gauss_newton_approx = False)
# #     Hmisfit = ReducedHessian(model, misfit_only=True)
# #     k = 80
# #     p = 20
# #     if rank == 0:
# #         print ("Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )

# #     # temp_petsc_vec_x_para = dlx.la.create_petsc_vector_wrap(x[hpx.PARAMETER])
# #     # Omega = MultiVector(temp_petsc_vec_x_para, k+p)
# #     # Bq,r = Omega.Borthogonalize(prior.R)

# #     # def petsc2array(v):
# #     #     s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
# #     #     return s
# #     temp_par_vector = dlx.la.create_petsc_vector_wrap(x[hpx.PARAMETER])
# #     # Omega = hpx.MultiVector(x[hpx.PARAMETER], k+p)
# #     Omega = hpx.MultiVector(temp_par_vector, k+p)
# #     hpx.parRandom.normal(1., Omega)
# #     Bq,r = Omega.Borthogonalize(prior.R)

# #     #validity: Omega.T @ Bq = I
# #             #  (nvec,m) @ (m,nvec)

# #     print(r)

# #     # print(r)
# #     # print(type(Bq))

# #     # # print(petsc2array(Bq))
# #     # print(r)
# #     # print(type(Bq))

# #     ###########ignore for now#############################


# #     #doublepass G stuff:
# #     # nvec = Omega.nvec
# #     # Ybar = MultiVector(Omega[0], nvec)
# #     # Q = Omega
# #     # for i in range(s):
# #     #     MatMvMult(A, Q, Ybar)
# #     #     MatMvMult(Solver2Operator(Binv), Ybar, Q)

# #     # Q = MultiVector(Omega)
# #     # Bq,r = Omega.Borthogonalize(prior.R)

# #     # print(r)


# # if __name__ == "__main__":
# #     nx = 64
# #     ny = 64
# #     noise_variance = 1e-6
# #     prior_param = {"gamma": 0.040, "delta": 0.8}
# #     mesh_filename = "../example/meshes/circle.xdmf"
# #     run_inversion(mesh_filename, nx, ny, noise_variance, prior_param)

# #     comm = MPI.COMM_WORLD
# #     # if comm.rank == 0:
# #     #     plt.savefig("qpact_result_FD_Gradient_Hessian_Check")
# #     #     plt.show()
