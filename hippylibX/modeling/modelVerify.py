import numpy as np

from .variables import STATE, PARAMETER, ADJOINT
# from .reducedHessian import ReducedHessian
from ..utils.random import parRandom
    
import dolfinx as dlx

# def modelVerify(model,m0, is_quadratic = False, misfit_only=False, verbose = True, eps = None):
def modelVerify(Vh, comm, model,m0, is_quadratic = False, misfit_only=False, verbose = True, eps = None):

    """
    Verify the reduced Gradient and the Hessian of a model.
    It will produce two loglog plots of the finite difference checks for the gradient and for the Hessian.
    It will also check for symmetry of the Hessian.
    """
    # print(comm.rank,":",verbose)

    if misfit_only:
        index = 2
    else:
        index = 0
    
    h = model.generate_vector(PARAMETER)
    # parRandom.normal(1., h)
    # parRandom(comm, 1., h) #supply comm to this method
    h.set(5.)

    # print(comm.rank,":",h.getArray())
    
    x = model.generate_vector()
    x[PARAMETER] = m0.vector

    # print(comm.rank,":",x[PARAMETER].min(),":",x[PARAMETER].max())

    #below will destroy values in x[PARAMETER], so have to preserve it before calling

    m_fun_true = dlx.fem.Function(Vh[PARAMETER])
    m_fun_true.x.array[:] = m0.x.array[:]

    model.solveFwd(x[STATE], x)
    
    x[PARAMETER] = m_fun_true.vector
    m0 = m0.vector

    # print(comm.rank,":",x[PARAMETER].min(),":",x[PARAMETER].max())
    
    model.solveAdj(x[ADJOINT], x ,Vh[ADJOINT])

    # print(comm.rank,":",x[ADJOINT].getArray())
    
    cx = model.cost(x)
    
    grad_x = model.generate_vector(PARAMETER)


    # print(x[STATE].min(),":",x[STATE].max())
    # print(x[PARAMETER].min(),":",x[PARAMETER].max())

    _,grad_x = model.evalGradientParameter(x,misfit_only=misfit_only)
    
    # grad_xh = grad_x.inner( h )
    
    # print(grad_x.min(),":",grad_x.max()) #(2992, -4729.582595240499) : (978, 162508.4367134074)

    grad_xh = grad_x.dot( h )
    # print(grad_xh) #27663.37064562618

    # model.setPointForHessianEvaluations(x)
    # H = ReducedHessian(model, misfit_only=misfit_only)
    # Hh = model.generate_vector(PARAMETER)
    # H.mult(h, Hh)
    

    if eps is None:
        n_eps = 32
        eps = np.power(.5, np.arange(n_eps))
        eps = eps[::-1]
    else:
        n_eps = eps.shape[0]
    
    err_grad = np.zeros(n_eps)
    err_H = np.zeros(n_eps)
    
    # print(comm.rank,":",m0.min(),":",m0.max())

    # for i in range(n_eps):
    
    for i in range(n_eps):
        my_eps = eps[i]
        
        x_plus = model.generate_vector()
        x_plus[PARAMETER].axpy(1., m0 )
        x_plus[PARAMETER].axpy(my_eps, h)

        # print(comm.rank,":",x_plus[STATE].min(),":",x_plus[STATE].max())    
        # print(comm.rank,":",x_plus[PARAMETER].min(),":",x_plus[PARAMETER].max())
        
        model.solveFwd(x_plus[STATE], x_plus)

        # print(comm.rank,":",x_plus[STATE].min(),":",x_plus[STATE].max())
        # print(comm.rank,":",x_plus[PARAMETER].min(),":",x_plus[PARAMETER].max())

        # model.solveAdj(x_plus[ADJOINT], x_plus)

        # print(comm.rank,":",x_plus[STATE].min(),":",x_plus[STATE].max())        
        # print(comm.rank,":",x_plus[PARAMETER].min(),":",x_plus[PARAMETER].max())
        
        dc = model.cost(x_plus)[index] - cx[index]
        
        err_grad[i] = abs(dc/my_eps - grad_xh)
        
        # Check the Hessian
        grad_xplus = model.generate_vector(PARAMETER)

    #    # model.evalGradientParameter(x_plus, grad_xplus,misfit_only=misfit_only)

    #     # err  = grad_xplus - grad_x
    #     # err *= 1./my_eps
    #     # err -= Hh
        
    #     # err_H[i] = err.norm('linf')
    # if(comm.rank == 0):
        # print(len(err_H))
        # print(len(eps),len(err_grad))
        # print(eps,'\n')
        # print(err_grad)
    # print(comm.rank, ":" , "hello-1")
    
    # if verbose: #true only for rank == 0
        # modelVerifyPlotErrors(comm, is_quadratic, eps, err_grad, err_H)

    # xx = model.generate_vector(PARAMETER)
    # parRandom.normal(1., xx)
    # yy = model.generate_vector(PARAMETER)
    # parRandom.normal(1., yy)
    
    # ytHx = H.inner(yy,xx)
    # xtHy = H.inner(xx,yy)
    # if np.abs(ytHx + xtHy) > 0.: 
    #     rel_symm_error = 2*abs(ytHx - xtHy)/(ytHx + xtHy)
    # else:
    #     rel_symm_error = abs(ytHx - xtHy)
    # if verbose:
    #     print( "(yy, H xx) - (xx, H yy) = ", rel_symm_error)
    #     if rel_symm_error > 1e-10:
    #         print( "HESSIAN IS NOT SYMMETRIC!!")
            
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
        plt.show()


        # plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        
        # plt.title("FD Gradient Check")
        # plt.subplot(122)
        # plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
        # plt.title("FD Hessian Check")