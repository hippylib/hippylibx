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
    # parRandom(comm).normal(1., h)
    h.array[:] = 5.

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


    test_func = vector2Function(Hh,model.misfit.Vh[PARAMETER])

    comm = model.misfit.mesh.comm

    V = test_func.function_space
    x_coords,y_coords = dlx.fem.Function(V), dlx.fem.Function(V)
    x_coords.interpolate(lambda x: x[0])
    y_coords.interpolate(lambda x: x[1] )
    
    x_coords_loc = x_coords.vector.array
    y_coords_loc = y_coords.vector.array

    u_at_vertices = dlx.fem.Function(V)
    u_at_vertices.interpolate(test_func)
    values_local = u_at_vertices.vector.array

    values_all = comm.gather([x_coords_loc, y_coords_loc, values_local],root=  0)

    if(comm.rank == 0):
        concat_data = np.concatenate(values_all, axis = 1)
        sorted_indices = np.lexsort((concat_data[1], concat_data[0]))
        sorted_data = [concat_data[:,i] for i in sorted_indices]
        sorted_func_vals = np.array(sorted_data)[:,-1]
        # print(type(sorted_func_vals))
        np.save(f'Hh_{comm.size}_procs.npy',sorted_func_vals)


    return

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
        
        x_plus[PARAMETER].array[:] = x_plus[PARAMETER].array + 1. * m0.array
        x_plus[PARAMETER].array[:] = x_plus[PARAMETER].array + my_eps * h.array


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
    parRandom(comm).normal(1., xx)
    yy = model.generate_vector(PARAMETER)
    parRandom(comm).normal(1., yy)
    
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

    return eps, err_grad, err_H


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
        plt.savefig("result_using_1_proc.png")
    else:  
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
        plt.title("FD Hessian Check")
        plt.savefig("result_using_1_proc.png")