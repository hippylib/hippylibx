# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import numpy as np
from .variables import STATE, PARAMETER, ADJOINT
from .reducedHessian import ReducedHessian
from ..utils.random import parRandom
import dolfinx as dlx
import petsc4py
from ..algorithms import linalg


def modelVerify(
    model,
    m0: dlx.la.Vector,
    is_quadratic=False,
    misfit_only=False,
    verbose=True,
    eps=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    parRandom.normal(1.0, h)
    x = model.generate_vector()

    x[PARAMETER] = m0
    model.solveFwd(x[STATE], x)
    model.solveAdj(x[ADJOINT], x)
    cx = model.cost(x)

    grad_x = model.generate_vector(PARAMETER)
    model.evalGradientParameter(x, grad_x, misfit_only=misfit_only)
    grad_xh = linalg.inner(grad_x, h)

    model.setPointForHessianEvaluations(x)

    H = ReducedHessian(model, misfit_only=misfit_only)

    Hh = model.generate_vector(PARAMETER)
    H.mat.mult(h.petsc_vec, Hh.petsc_vec)

    if eps is None:
        n_eps = 32
        eps = np.power(0.5, np.arange(n_eps))
        eps = eps[::-1]
    else:
        n_eps = eps.shape[0]
    err_grad = np.zeros(n_eps)
    err_H = np.zeros(n_eps)

    for i in range(n_eps):
        my_eps = eps[i]
        x_plus = model.generate_vector()

        x_plus[PARAMETER].array[:] = m0.array + my_eps * h.array
        model.solveFwd(x_plus[STATE], x_plus)
        model.solveAdj(x_plus[ADJOINT], x_plus)

        dc = model.cost(x_plus)[index] - cx[index]

        err_grad[i] = abs(dc / my_eps - grad_xh)

        # Check the Hessian
        grad_xplus = model.generate_vector(PARAMETER)
        model.evalGradientParameter(x_plus, grad_xplus, misfit_only=misfit_only)

        err = grad_xplus.petsc_vec - grad_x.petsc_vec

        err.scale(1.0 / my_eps)
        err.axpy(-1.0, Hh.petsc_vec)

        err_H[i] = err.norm(petsc4py.PETSc.NormType.NORM_INFINITY)

        err.destroy()
    if verbose:
        modelVerifyPlotErrors(is_quadratic, eps, err_grad, err_H)

    xx = model.generate_vector(PARAMETER)
    parRandom.normal(1.0, xx)
    yy = model.generate_vector(PARAMETER)
    parRandom.normal(1.0, yy)

    #######################################
    Ay = model.generate_vector(PARAMETER)
    Ay.petsc_vec.scale(0.0)
    H.mat.mult(xx.petsc_vec, Ay.petsc_vec)
    ytHx = yy.petsc_vec.dot(Ay.petsc_vec)

    Ay.petsc_vec.scale(0.0)
    H.mat.mult(yy.petsc_vec, Ay.petsc_vec)
    xtHy = xx.petsc_vec.dot(Ay.petsc_vec)

    # # #######################################

    if np.abs(ytHx + xtHy) > 0.0:
        rel_symm_error = 2 * abs(ytHx - xtHy) / (ytHx + xtHy)
    else:
        rel_symm_error = abs(ytHx - xtHy)
    if verbose:
        print("(yy, H xx) - (xx, H yy) = ", rel_symm_error)
        if rel_symm_error > 1e-10:
            print("HESSIAN IS NOT SYMMETRIC!!")
    return {
        "eps": eps,
        "err_grad": err_grad,
        "err_H": err_H,
        "sym_Hessian_value": rel_symm_error,
    }


def modelVerifyPlotErrors(
    is_quadratic: bool, eps: np.ndarray, err_grad: np.ndarray, err_H: np.ndarray
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Matplotlib is not installed.")
        return
    if is_quadratic:
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps * (err_grad[0] / eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(
            eps[0],
            err_H[0],
            "-ob",
            [10 * eps[0], eps[0], 0.1 * eps[0]],
            [err_H[0], err_H[0], err_H[0]],
            "-.k",
        )
        plt.title("FD Hessian Check")
    else:
        plt.figure()
        plt.subplot(121)
        plt.loglog(eps, err_grad, "-ob", eps, eps * (err_grad[0] / eps[0]), "-.k")
        plt.title("FD Gradient Check")
        plt.subplot(122)
        plt.loglog(eps, err_H, "-ob", eps, eps * (err_H[0] / eps[0]), "-.k")
        plt.title("FD Hessian Check")
