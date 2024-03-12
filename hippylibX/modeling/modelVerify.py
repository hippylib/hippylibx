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
    h.array[:] = 0.2
    # parRandom.normal(1., h)
    # parRandom(comm).normal(1., h)

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

    #need to check mult operation for why "Hessian not symm" for
    # multiprocessor runs

    test_func = vector2Function(Hh,model.misfit.Vh[PARAMETER])

    V = test_func.function_space

    x_coords,y_coords = dlx.fem.Function(V), dlx.fem.Function(V)
    x_coords.interpolate(lambda x: x[0])
    y_coords.interpolate(lambda x: x[1] )
    
    x_coords_loc = x_coords.vector.array
    y_coords_loc = y_coords.vector.array

    u_at_vertices = dlx.fem.Function(V)
    u_at_vertices.interpolate(test_func)
    values_loc = u_at_vertices.vector.array

    loc_vals_comb = [x_coords_loc, y_coords_loc, values_loc]

    all_vals_comb = comm.gather(loc_vals_comb, root = 0)


    if(comm.rank == 0):
        all_x, all_y, all_vals =  [],[],[]

        for sublist in all_vals_comb:
            all_x.append(sublist[0])
            all_y.append(sublist[1])
            all_vals.append(sublist[2])
            
        all_x_flat = np.concatenate([arr.flatten() for arr in all_x])
        all_y_flat = np.concatenate([arr.flatten() for arr in all_y])
        all_vals_flat = np.concatenate([arr.flatten() for arr in all_vals])
    
        combined_tuple_version = [(all_x_flat[i], all_y_flat[i], all_vals_flat[i]) for i in range(len(all_x_flat))]

        sorted_combined_tuple_version = sorted(combined_tuple_version, key = lambda x: (x[0], x[1]) )

        x_coords_final, y_coords_final, values_final = [], [], []

        for j in range(len(sorted_combined_tuple_version)):
            x_coords_final.append(sorted_combined_tuple_version[j][0])
            y_coords_final.append(sorted_combined_tuple_version[j][1])
            values_final.append(sorted_combined_tuple_version[j][2])
        
        np.savetxt(f'x_coords_X_{comm.size}_procs_v3',x_coords_final)
        np.savetxt(f'y_coords_X_{comm.size}_procs_v3',y_coords_final) 
        np.savetxt(f'Hh_{comm.size}_procs_v3',values_final)

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
    parRandom.normal(1., xx)
    # parRandom(comm).normal(1., xx)
    
    yy = model.generate_vector(PARAMETER)
    parRandom.normal(1., yy)
    # parRandom(comm).normal(1., yy)
    
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
        plt.savefig("result_using_4_proc_v3.png")
    else:  
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps*(err_grad[0]/eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps, err_H, "-ob", eps, eps*(err_H[0]/eps[0]), "-.k")
        plt.title("FD Hessian Check")
        plt.savefig("result_using_4_proc_v3.png")