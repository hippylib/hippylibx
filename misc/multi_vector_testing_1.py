import sys
import os
import numpy as np
from mpi4py import MPI
import dolfinx as dlx
import ufl
import petsc4py
import dolfinx.fem.petsc

sys.path.append(os.path.abspath("../"))
import hippylibX as hpx
sys.path.append(os.path.abspath("../example"))
from hippylibX.modeling.reducedHessian import ReducedHessian

class MultiVector:
    def __init__(self,example_vec: petsc4py.PETSc.Vec, nvec:int):
        self.data = []
        self.nvec = nvec

        for i in range(self.nvec):
            self.data.append(example_vec.duplicate(example_vec.getArray()))

    def __getitem__(self,k):
        return self.data[k]

    def scale(self,k : int, a: float):
        self[k].scale(a)

    def dot_v(self, v : petsc4py.PETSc.Vec) -> np.array:
        return_values = []
        for i in range(self.nvec):
            return_values.append(self[i].dot(v)  )            
        return np.array(return_values)

    def reduce(self, alpha: np.array) -> petsc4py.PETSc.Vec:
        return_vec = self[0].duplicate()
        for i in range(self.nvec):
            return_vec.axpy(alpha[i],self[i])
        return return_vec   
    
    def Borthogonalize(self,B):
        """ 
        Returns :math:`QR` decomposition of self. :math:`Q` and :math:`R` satisfy the following relations in exact arithmetic

        .. math::
            R \\,= \\,Z, && (1),

            Q^*BQ\\, = \\, I, && (2),

            Q^*BZ \\, = \\,R, && (3),

            ZR^{-1} \\, = \\, Q, && (4). 
        
        Returns:

            :code:`Bq` of type :code:`MultiVector` -> :code:`B`:math:`^{-1}`-orthogonal vectors
            :code:`r` of type :code:`ndarray` -> The :math:`r` of the QR decomposition.
        
        .. note:: :code:`self` is overwritten by :math:`Q`.    
        """
        return self._mgs_stable(B)

    def _mgs_stable(self, B : petsc4py.PETSc.Mat):
        """ 
        Returns :math:`QR` decomposition of self, which satisfies conditions (1)--(4).
        Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
        for computing the :math:`B`-orthogonal :math:`QR` factorization.
        
        References:
            1. `A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized \
            Hermitian Eigenvalue Problems with application to computing \
            Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885`
            2. `W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980`
        
        https://github.com/arvindks/kle
        
        """
        n = self.nvec
        Bq = MultiVector(self[0], n) 
        r  = np.zeros((n,n), dtype = 'd')
        reorth = np.zeros((n,), dtype = 'd')
        eps = np.finfo(np.float64).eps
        
        for k in np.arange(n):
            B.mult(self[k], Bq[k])
            t = np.sqrt( Bq[k].dot(self[k]))
            
            nach = 1;    u = 0;
            while nach:
                u += 1
                for i in np.arange(k):
                    s = Bq[i].dot(self[k])
                    r[i,k] += s
                    self[k].axpy(-s, self[i])
                    
                B.mult(self[k], Bq[k])
                tt = np.sqrt(Bq[k].dot(self[k]))
                if tt > t*10.*eps and tt < t/10.:
                    nach = 1;    t = tt;
                else:
                    nach = 0;
                    if tt < 10.*eps*t:
                        tt = 0.
            

            reorth[k] = u
            r[k,k] = tt
            if np.abs(tt*eps) > 0.:
                tt = 1./tt
            else:
                tt = 0.
                
            self.scale(k, tt)
            Bq.scale(k, tt)
            
        return Bq, r 

def MatMvMult(A, x, y):
    assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
    if hasattr(A,'matMvMult'):
        A.matMvMult(x,y)
    else:
        for i in range(x.nvec()):
            A.mult(x[i], y[i])

def MatMvTranspmult(A, x, y):
    assert x.nvec() == y.nvec(), "x and y have non-matching number of vectors"
    assert hasattr(A,'transpmult'), "A does not have transpmult method implemented"
    if hasattr(A,'matMvTranspmult'):
        A.matMvTranspmult(x,y)
    else:
        for i in range(x.nvec()):
            A.transpmult(x[i], y[i])
        
def MvDSmatMult(X, A, Y):
    assert X.nvec() == A.shape[0], "X Number of vecs incompatible with number of rows in A"
    assert Y.nvec() == A.shape[1], "Y Number of vecs incompatible with number of cols in A"
    for j in range(Y.nvec()):
        Y[j].zero()
        X.reduce(Y[j], A[:,j].flatten())


class Poisson_Approximation:
    def __init__(self, f: float):
        self.f = f
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})

    def __call__(
        self, u: dlx.fem.Function, m: dlx.fem.Function, p: dlx.fem.Function
    ) -> ufl.form.Form:
        return (
            ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * self.dx
            - self.f * p * self.dx
        )


class PoissonMisfitForm:
    def __init__(self, d: float, sigma2: float):
        self.d = d
        self.sigma2 = sigma2
        self.dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})

    def __call__(self, u: dlx.fem.Function, m: dlx.fem.Function) -> ufl.form.Form:
        return 0.5 / self.sigma2 * ufl.inner(u - self.d, u - self.d) * self.dx


def master_print(comm, *args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs)


def run_inversion(nx: int, ny: int, noise_variance: float, prior_param: dict) -> None:
    sep = "\n" + "#" * 80 + "\n"
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.size

    msh = dlx.mesh.create_unit_square(comm, nx, ny, dlx.mesh.CellType.quadrilateral)
    Vh_phi = dlx.fem.FunctionSpace(msh, ("Lagrange", 2))
    Vh_m = dlx.fem.FunctionSpace(msh, ("Lagrange", 1))
    Vh = [Vh_phi, Vh_m, Vh_phi]
    ndofs = [
        Vh_phi.dofmap.index_map.size_global * Vh_phi.dofmap.index_map_bs,
        Vh_m.dofmap.index_map.size_global * Vh_m.dofmap.index_map_bs,
    ]
    master_print(comm, sep, "Set up the mesh and finite element spaces", sep)
    master_print(comm, "Number of dofs: STATE={0}, PARAMETER={1}".format(*ndofs))

    # dirichlet B.C.
    uD = dlx.fem.Function(Vh[hpx.STATE])
    uD.interpolate(lambda x: x[1])
    uD.x.scatter_forward()

    def top_bottom_boundary(x):
        return np.logical_or(np.isclose(x[1], 1), np.isclose(x[1], 0))

    fdim = msh.topology.dim - 1
    top_bottom_boundary_facets = dlx.mesh.locate_entities_boundary(
        msh, fdim, top_bottom_boundary
    )
    dirichlet_dofs = dlx.fem.locate_dofs_topological(
        Vh[hpx.STATE], fdim, top_bottom_boundary_facets
    )
    bc = dlx.fem.dirichletbc(uD, dirichlet_dofs)

    # bc0
    uD_0 = dlx.fem.Function(Vh[hpx.STATE])
    uD_0.interpolate(lambda x: 0.0 * x[0])
    uD_0.x.scatter_forward()
    bc0 = dlx.fem.dirichletbc(uD_0, dirichlet_dofs)

    # # FORWARD MODEL
    f = dlx.fem.Constant(msh, dlx.default_scalar_type(0.0))
    pde_handler = Poisson_Approximation(f)
    pde = hpx.PDEVariationalProblem(Vh, pde_handler, [bc], [bc0], is_fwd_linear=True)

    # GROUND TRUTH
    m_true = dlx.fem.Function(Vh_m)
    m_true.interpolate(
        lambda x: np.log(2 + 7 * (((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) ** 0.5 > 0.2))
    )
    m_true.x.scatter_forward()

    m_true = m_true.x
    u_true = pde.generate_state()
    x_true = [u_true, m_true, None]
    pde.solveFwd(u_true, x_true)

    # # LIKELIHOOD
    d = dlx.fem.Function(Vh[hpx.STATE])
    d.x.array[:] = u_true.array[:]
    hpx.parRandom.normal_perturb(np.sqrt(noise_variance), d.x)
    d.x.scatter_forward()
    misfit_form = PoissonMisfitForm(d, noise_variance)
    misfit = hpx.NonGaussianContinuousMisfit(Vh, misfit_form, [bc0])
    prior_mean = dlx.fem.Function(Vh_m)
    prior_mean.x.array[:] = 0.01
    prior_mean = prior_mean.x

    prior = hpx.BiLaplacianPrior(
        Vh_m, prior_param["gamma"], prior_param["delta"], mean=prior_mean
    )
    model = hpx.Model(pde, prior, misfit)

    noise = prior.generate_parameter("noise")
    m0 = prior.generate_parameter(0)
    hpx.parRandom.normal(1.0, noise)
    prior.sample(noise, m0)


    initial_guess_m = prior.generate_parameter(0)
    initial_guess_m.array[:] = prior_mean.array[:]

    x = [
        model.generate_vector(hpx.STATE),
        initial_guess_m,
        model.generate_vector(hpx.ADJOINT),
    ]
    if rank == 0:
        print(sep, "Find the MAP point", sep)

    parameters = hpx.ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = 1e-6
    parameters["abs_tolerance"] = 1e-9
    parameters["max_iter"] = 500
    parameters["cg_coarse_tolerance"] = 5e-1
    parameters["globalization"] = "LS"
    parameters["GN_iter"] = 20
    if rank != 0:
        parameters["print_level"] = -1

    solver = hpx.ReducedSpaceNewtonCG(model, parameters)

    x = solver.solve(x)

    m_fun = hpx.vector2Function(x[hpx.PARAMETER], Vh[hpx.PARAMETER], name="m_map")
    m_true_fun = hpx.vector2Function(m_true, Vh[hpx.PARAMETER], name="m_true")

    V_P1 = dlx.fem.FunctionSpace(msh, ("Lagrange", 1))

    u_true_fun = dlx.fem.Function(V_P1, name="u_true")
    u_true_fun.interpolate(hpx.vector2Function(u_true, Vh[hpx.STATE]))
    u_true_fun.x.scatter_forward()

    u_map_fun = dlx.fem.Function(V_P1, name="u_map")
    u_map_fun.interpolate(hpx.vector2Function(x[hpx.STATE], Vh[hpx.STATE]))
    u_map_fun.x.scatter_forward()

    d_fun = dlx.fem.Function(V_P1, name="data")
    d_fun.interpolate(d)
    d_fun.x.scatter_forward()

    with dlx.io.VTXWriter(
        msh.comm,
        "poisson_Dirichlet_BiLaplacian_prior_np{0:d}_Prior.bp".format(nproc),
        [m_fun, m_true_fun, u_map_fun, u_true_fun, d_fun],
    ) as vtx:
        vtx.write(0.0)

    if solver.converged:
        master_print(comm, "\nConverged in ", solver.it, " iterations.")
    else:
        master_print(comm, "\nNot Converged")
    master_print(
        comm, "Termination reason: ", solver.termination_reasons[solver.reason]
    )
    master_print(comm, "Final gradient norm: ", solver.final_grad_norm)
    master_print(comm, "Final cost: ", solver.final_cost)

    optimizer_results = {}
    if (
        solver.termination_reasons[solver.reason]
        == "Norm of the gradient less than tolerance"
    ):
        optimizer_results["optimizer"] = True
    else:
        optimizer_results["optimizer"] = False


    model.setPointForHessianEvaluations(x, gauss_newton_approx = False)
    Hmisfit = ReducedHessian(model, misfit_only=True)
    k = 80
    p = 20
    if rank == 0:
        print ("Double Pass Algorithm. Requested eigenvectors: {0}; Oversampling {1}.".format(k,p) )
    
    temp_petsc_vec_x_para = dlx.la.create_petsc_vector_wrap(x[hpx.PARAMETER])
    Omega = MultiVector(temp_petsc_vec_x_para, k+p)
    Bq,r = Omega.Borthogonalize(prior.R)

    def petsc2array(v):
        s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
        return s


    # print(petsc2array(Bq))
    print(r)

    ###########ignore for now#############################


    #doublepass G stuff:
    # nvec = Omega.nvec
    # Ybar = MultiVector(Omega[0], nvec)
    # Q = Omega
    # for i in range(s):
    #     MatMvMult(A, Q, Ybar)
    #     MatMvMult(Solver2Operator(Binv), Ybar, Q)
    
    # Q = MultiVector(Omega)
    # Bq,r = Omega.Borthogonalize(prior.R)

    # print(r)
    


if __name__ == "__main__":
    nx = 64
    ny = 64
    noise_variance = 1e-4
    prior_param = {"gamma": 0.03, "delta": 0.3}
    run_inversion(nx, ny, noise_variance, prior_param)

    comm = MPI.COMM_WORLD
    # if comm.rank == 0:
    #     plt.savefig("poisson_result_FD_Gradient_Hessian_Check")
    #     plt.show()
