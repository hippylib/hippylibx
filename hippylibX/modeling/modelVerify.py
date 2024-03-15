import numpy as np

from .variables import STATE, PARAMETER, ADJOINT
from .reducedHessian import ReducedHessian
from ..utils.random import parRandom
    
import dolfinx as dlx
import petsc4py
import mpi4py

from ..utils import vector2Function
from ..algorithms import linalg

def modelVerify(comm : mpi4py.MPI.Intracomm, model, m0 : dlx.la.Vector, is_quadratic = False, misfit_only=False, verbose = True, eps = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Verify the reduced Gradient and the Hessian of a model.
    It will produce two loglog plots of the finite difference checks for the gradient and for the Hessian.
    It will also check for symmetry of the Hessian.
    """

    if misfit_only:
        index = 2
    else:
        index = 0
    
    h = model.generate_vector(PARAMETER)

    parRandom.normal(1., h)


    x = model.generate_vector()
    
    x[PARAMETER] = m0
    model.solveFwd(x[STATE], x)

    model.solveAdj(x[ADJOINT], x)

    cx = model.cost(x)
    
    grad_x = model.generate_vector(PARAMETER)
    model.evalGradientParameter(x,grad_x, misfit_only=misfit_only)   

    grad_xh = linalg.inner(grad_x, h)


    temp_petsc_vec_grad_x = dlx.la.create_petsc_vector_wrap(grad_x)

    model.setPointForHessianEvaluations(x)
 
    H = ReducedHessian(model, misfit_only=misfit_only)
    Hh = model.generate_vector(PARAMETER)
    H.mult(h, Hh)

    if eps is None:
        n_eps = 32
        eps = np.power(.5, np.arange(n_eps))
        eps = eps[::-1]
    else:
        n_eps = eps.shape[0]

    err_grad = np.zeros(n_eps)
    err_H = np.zeros(n_eps)
            
    for i in range(n_eps):
        my_eps = eps[i]
        x_plus = model.generate_vector()
                
        x_plus[PARAMETER].array[:]  = m0.array + my_eps * h.array

        model.solveFwd(x_plus[STATE], x_plus)

        model.solveAdj(x_plus[ADJOINT], x_plus)

        dc = model.cost(x_plus)[index] - cx[index]
        
        err_grad[i] = abs(dc/my_eps - grad_xh)
        
        # Check the Hessian
        grad_xplus = model.generate_vector(PARAMETER)
        model.evalGradientParameter(x_plus, grad_xplus,misfit_only=misfit_only)
        
        temp_petsc_vec_grad_xplus = dlx.la.create_petsc_vector_wrap(grad_xplus)
        
        err = temp_petsc_vec_grad_xplus - temp_petsc_vec_grad_x
        temp_petsc_vec_grad_xplus.destroy()

        err.scale(1./my_eps)
        temp_petsc_vec_Hh = dlx.la.create_petsc_vector_wrap(Hh)
        err.axpy(-1., temp_petsc_vec_Hh)
        temp_petsc_vec_Hh.destroy()

        err_H[i] = err.norm(petsc4py.PETSc.NormType.NORM_INFINITY)
        err.destroy()

    # if verbose:
        # modelVerifyPlotErrors(comm, misfit_only,is_quadratic, eps, err_grad, err_H)
        #comm and misfit_only are being passed for plotting purposes only-
        #the title of the plot for saved figures.
        #has to be removed for the final version.

    temp_petsc_vec_grad_x.destroy()
    
    xx = model.generate_vector(PARAMETER)
    parRandom.normal(1., xx)


    yy = model.generate_vector(PARAMETER)
    parRandom.normal(1., yy)

    ytHx = H.inner(yy,xx)

    xtHy = H.inner(xx,yy)

    if np.abs(ytHx + xtHy) > 0.: 
        rel_symm_error = 2*abs(ytHx - xtHy)/(ytHx + xtHy)
    else:
        rel_symm_error = abs(ytHx - xtHy)


    if verbose:
        print( "(yy, H xx) - (xx, H yy) = ", rel_symm_error)
        if rel_symm_error > 1e-10:
            print( "HESSIAN IS NOT SYMMETRIC!!")


    #clean-up ops
    # if(model.problem.A is not None):
    #     model.problem.A.destroy()
    #     model.problem.A = None
        
    
    # if(model.problem.At is not None):
    #     model.problem.At.destroy()
    #     model.problem.At = None
    
    # if(model.problem.C is not None):
    #     model.problem.C.destroy()
    #     model.problem.C = None
    

    # if(model.problem.Wuu is not None):
    #     model.problem.Wuu.destroy()
    #     model.problem.Wuu = None
    

    # if(model.problem.Wmu is not None):
    #     model.problem.Wmu.destroy()
    #     model.problem.Wmu = None


    # if(model.problem.Wum is not None):
    #     model.problem.Wum.destroy()
    #     model.problem.Wum = None


    # if(model.problem.Wmm is not None):
    #     model.problem.Wmm.destroy()
    #     model.problem.Wmm = None



    return eps, err_grad, err_H, rel_symm_error


def modelVerifyPlotErrors(comm, misfit_only, is_quadratic : bool, eps : np.ndarray, err_grad : np.ndarray, err_H : np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except:
        print( "Matplotlib is not installed.")
        return
    if is_quadratic:
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps[0], err_H[0], "-ob", [10*eps[0], eps[0], 0.1*eps[0]], [err_H[0],err_H[0],err_H[0]], "-.k")
        plt.title("FD Hessian Check")
        plt.savefig(f"result_with_misfit_{misfit_only}_using_{comm.size}_procs.png")
    else:  
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
        plt.title("FD Hessian Check")
        plt.savefig(f"result_with_misfit_{misfit_only}_using_{comm.size}_procs.png")
