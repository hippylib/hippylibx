import numpy as np
from .variables import STATE, PARAMETER, ADJOINT
from .reducedHessian import ReducedHessian
from ..utils.random import parRandom
    
import dolfinx as dlx
import petsc4py
import mpi4py
from ..utils import vector2Function
from ..algorithms import linalg
def modelVerify(model, m0 : dlx.la.Vector, is_quadratic = False, misfit_only=False, verbose = True, eps = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    H_object = ReducedHessian(model, misfit_only=misfit_only)
    H = H_object.as_petsc_wrapper()


    # H_obj = ReducedHessian(model, misfit_only=misfit_only)
    # H  = petsc4py.PETSc.Mat().createPython(model.prior.M.getSizes(),comm=model.prior.Vh.mesh.comm)
    # H.setPythonContext(H_obj)
    # H.setUp()

    Hh = model.generate_vector(PARAMETER)
    temp_petsc_vec_h = dlx.la.create_petsc_vector_wrap(h)
    temp_petsc_vec_Hh = dlx.la.create_petsc_vector_wrap(Hh)

    H.mult(temp_petsc_vec_h, temp_petsc_vec_Hh)
    # H.mult(h,Hh)
    temp_petsc_vec_h.destroy()
    temp_petsc_vec_Hh.destroy()

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
    if verbose:
        modelVerifyPlotErrors(is_quadratic, eps, err_grad, err_H)
       
     
    temp_petsc_vec_grad_x.destroy()
    
    xx = model.generate_vector(PARAMETER)
    parRandom.normal(1., xx)
    yy = model.generate_vector(PARAMETER)
    parRandom.normal(1., yy)

    # ytHx = H.inner(yy,xx)
    # xtHy = H.inner(xx,yy)

    #######################################
    temp_petsc_vec_xx = dlx.la.create_petsc_vector_wrap(xx)
    temp_petsc_vec_yy = dlx.la.create_petsc_vector_wrap(yy)

    # #in two separate operations:
    Ay = model.generate_vector(PARAMETER)
    temp_petsc_vec_Ay = dlx.la.create_petsc_vector_wrap(Ay)
    temp_petsc_vec_Ay.scale(0.)
    H.mult(temp_petsc_vec_xx,temp_petsc_vec_Ay)
    ytHx = temp_petsc_vec_yy.dot(temp_petsc_vec_Ay)
    
    temp_petsc_vec_Ay.scale(0.)
    H.mult(temp_petsc_vec_yy, temp_petsc_vec_Ay)
    xtHy = temp_petsc_vec_xx.dot(temp_petsc_vec_Ay)    
    temp_petsc_vec_Ay.destroy()
    temp_petsc_vec_xx.destroy()
    temp_petsc_vec_yy.destroy()
    # #######################################


    if np.abs(ytHx + xtHy) > 0.: 
        rel_symm_error = 2*abs(ytHx - xtHy)/(ytHx + xtHy)
    else:
        rel_symm_error = abs(ytHx - xtHy)
    if verbose:
        print( "(yy, H xx) - (xx, H yy) = ", rel_symm_error)
        if rel_symm_error > 1e-10:
            print( "HESSIAN IS NOT SYMMETRIC!!")
    return {"eps":eps,"err_grad":err_grad, "err_H": err_H, "sym_Hessian_value":rel_symm_error}


def modelVerifyPlotErrors(is_quadratic : bool, eps : np.ndarray, err_grad : np.ndarray, err_H : np.ndarray) -> None:
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
    else:  
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
        plt.title("FD Hessian Check")