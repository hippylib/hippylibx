import numpy as np

from .variables import STATE, PARAMETER, ADJOINT
# from .reducedHessian import ReducedHessian
from ..utils.random import parRandom
    
import dolfinx as dlx
import petsc4py

def modelVerify(Vh, comm, model,m0, is_quadratic = False, misfit_only=False, verbose = True, eps = None):

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
    parRandom(comm).normal(1., h)
    # h.array[:] = 1.

    x = model.generate_vector()
    
    x[PARAMETER] = m0

    model.solveFwd(x[STATE], x)
        
    model.solveAdj(x[ADJOINT], x)
    
    cx = model.cost(x)
    
    grad_x = model.generate_vector(PARAMETER)
    model.evalGradientParameter(x,grad_x, misfit_only=misfit_only)
    
    grad_xh = dlx.la.create_petsc_vector_wrap(grad_x).dot( dlx.la.create_petsc_vector_wrap(h) )
    
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
        
        dlx.la.create_petsc_vector_wrap(x_plus[PARAMETER]).axpy(1.,dlx.la.create_petsc_vector_wrap(m0))
    
        dlx.la.create_petsc_vector_wrap(x_plus[PARAMETER]).axpy(my_eps,dlx.la.create_petsc_vector_wrap(h))
    
        
        model.solveFwd(x_plus[STATE], x_plus)

        model.solveAdj(x_plus[ADJOINT], x_plus)
        
        dc = model.cost(x_plus)[index] - cx[index]
        
        err_grad[i] = abs(dc/my_eps - grad_xh)
        
        # Check the Hessian
        grad_xplus = model.generate_vector(PARAMETER)
            
    if verbose: #true only for rank == 0
        modelVerifyPlotErrors(comm, is_quadratic, eps, err_grad, err_H)

            
    return eps, err_grad, err_H


def modelVerifyPlotErrors(comm, is_quadratic, eps, err_grad, err_H):
    try:
        import matplotlib.pyplot as plt
    except:
        print( "Matplotlib is not installed.")
        return
    
    if is_quadratic:
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps*( err_grad[0]/eps[0] ), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps[0], err_H[0], "-ob", [10*eps[0], eps[0], 0.1*eps[0]], [err_H[0],err_H[0],err_H[0]], "-.k")
        plt.title("FD Hessian Check")
    
    else:  
        # print(comm.rank, ":" , "hello")
        scale_val = err_grad[0]/eps[0]
        second_val = [value*scale_val for value in eps]
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, second_val, "-.k")
        plt.title("FD Gradient Check")
        # plt.ion()
        plt.show() #uncommenting this will return control to the user
        
        
        # plt.pause(0.001)
        
        # plt.show(block = False)
        # plt.pause(5)
        # plt.close()


        # plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        
        # plt.title("FD Gradient Check")
        # plt.subplot(122)
        # plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
        # plt.title("FD Hessian Check")
