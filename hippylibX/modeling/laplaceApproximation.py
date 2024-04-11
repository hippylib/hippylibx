# Copyright (c) 2016-2018, The University of Texas at Austin
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

from ..algorithms.lowRankOperator import LowRankOperator
from ..algorithms.multivector import MultiVector
import numpy as np
import petsc4py
import dolfinx as dlx


def not_implemented(func):
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"{func.__name__} Function has not been implemented.")

    return wrapper


class LowRankHessian:
    """
    Operator that represents the action of the low rank approximation
    of the Hessian and of its inverse.
    """

    def __init__(
        self,
        prior,
        d: np.array,
        U: MultiVector,
        createVecLeft=None,
        createVecRight=None,
    ):
        self.U = U
        self.prior = prior
        self.LowRankH = LowRankOperator(d, self.U)
        dsolve = d / (np.ones(d.shape, dtype=d.dtype) + d)
        self.LowRankHinv = LowRankOperator(dsolve, self.U)

        self.createVecLeft = createVecLeft
        self.createVecRight = createVecRight

        self.help = self.prior.R.createVecRight()
        self.help1 = self.prior.R.createVecLeft()

    def __del__(self) -> None:
        self.help.destroy()
        self.help1.destroy()

    def mult(self, x: petsc4py.PETSc.Vec, y: petsc4py.PETSc.Vec) -> None:
        self.prior.R.mult(x, y)
        self.LowRankH.mult(y, self.help)
        self.prior.R.mult(self.help, self.help1)
        y.axpy(1, self.help1)

    def solve(self, rhs: petsc4py.PETSc.Vec, sol: petsc4py.PETSc.Vec) -> None:
        self.prior.Rsolver.solve(rhs, sol)
        self.LowRankHinv.mult(rhs, self.help)
        sol.axpy(-1, self.help)


class LowRankPosteriorSampler:
    """
    Object to sample from the low rank approximation
    of the posterior.

        .. math:: y = ( I - U S U^{T}) x,

    where

    :math:`S = I - (I + D)^{-1/2}, x \\sim \\mathcal{N}(0, R^{-1}).`
    """

    def __init__(
        self,
        prior,
        d: np.array,
        U: MultiVector,
        createVecLeft=None,
        createVecRight=None,
    ):
        self.U = U
        self.prior = prior
        self.createVecLeft = createVecLeft
        self.createVecRight = createVecRight

        ones = np.ones(d.shape, dtype=d.dtype)
        self.d = ones - np.power(ones + d, -0.5)
        self.lrsqrt = LowRankOperator(self.d, self.U)
        self.help = self.prior.R.createVecLeft()

    def __del__(self) -> None:
        self.help.destroy()

    def sample(self, noise: dlx.la.Vector, s: dlx.la.Vector):
        temp_petsc_vec_noise = dlx.la.create_petsc_vector_wrap(noise)
        temp_petsc_vec_s = dlx.la.create_petsc_vector_wrap(s)

        self.prior.R.mult(temp_petsc_vec_noise, self.help)
        self.lrsqrt.mult(self.help, temp_petsc_vec_s)
        temp_petsc_vec_s.axpy(-1.0, temp_petsc_vec_noise)
        temp_petsc_vec_s.scale(-1.0)
        temp_petsc_vec_noise.destroy()
        temp_petsc_vec_s.destroy()


class LaplaceApproximator:
    """
    Class for the low rank Gaussian Approximation of the Posterior.
    This class provides functionality for approximate Hessian
    apply, solve, and Gaussian sampling based on the low rank
    factorization of the Hessian.

    In particular if :math:`d` and :math:`U` are the dominant eigenpairs of
    :math:`H_{\\mbox{misfit}} U[:,i] = d[i] R U[:,i]`
    then we have:

    - low rank Hessian apply: :math:`y = ( R + RU D U^{T}) x.`
    - low rank Hessian solve: :math:`y = (R^-1 - U (I + D^{-1})^{-1} U^T) x.`
    - low rank Hessian Gaussian sampling: :math:`y = ( I - U S U^{T}) x`, where :math:`S = I - (I + D)^{-1/2}` and :math:`x \\sim \\mathcal{N}(0, R^{-1}).`
    """

    def __init__(self, prior, d: np.array, U: MultiVector, mean=None):
        """
        Construct the Gaussian approximation of the posterior.
        Input:
        - :code:`prior`: the prior mode.
        - :code:`d`:     the dominant generalized eigenvalues of the Hessian misfit.
        - :code:`U`:     the dominant generalized eigenvector of the Hessian misfit :math:`U^T R U = I.`
        - :code:`mean`:  the MAP point.
        """
        self.prior = prior
        self.d = d
        self.U = U
        self.Hlr = LowRankHessian(prior, d, U)
        self.sampler = LowRankPosteriorSampler(self.prior, self.d, self.U)
        self.mean = None

    def cost(self, m: dlx.la.Vector) -> float:
        if self.mean is None:
            dm = m
        else:
            dm = m - self.mean
        temp_petsc_vec_dm = dlx.la.create_petsc_vector_wrap(dm)
        self.Hlr.mult(temp_petsc_vec_dm, self.Hlr.help1)
        return_value = 0.5 * self.Hlr.help1.dot(temp_petsc_vec_dm)
        temp_petsc_vec_dm.destroy()
        return return_value

    def sample(self, *args, **kwargs):
        """
        possible calls:

        1) :code:`sample(s_prior, s_post, add_mean=True)`

           Given a prior sample  :code:`s_prior` compute a sample :code:`s_post` from the posterior.

           - :code:`s_prior` is a sample from the prior centered at 0 (input).
           - :code:`s_post` is a sample from the posterior (output).
           - if :code:`add_mean=True` (default) then the samples will be centered at the map point.

        2) :code:`sample(noise, s_prior, s_post, add_mean=True)`

           Given :code:`noise` :math:`\\sim \\mathcal{N}(0, I)` compute a sample :code:`s_prior` from the prior and
           :code:`s_post` from the posterior.

           - :code:`noise` is a realization of white noise (input).
           - :code:`s_prior` is a sample from the prior (output).
           - :code:`s_post`  is a sample from the posterior.
           - if :code:`add_mean=True` (default) then the prior and posterior samples will be centered at the respective means.

        """
        add_mean = True
        for name, value in kwargs.items():
            if name == "add_mean":
                add_mean = value
            else:
                raise NameError(name)

        if len(args) == 2:
            self._sample_given_prior(args[0], args[1])
            if add_mean:
                args[1].axpy(1.0, self.mean)
        elif len(args) == 3:
            self._sample_given_white_noise(args[0], args[1], args[2])
            if add_mean:
                args[1].axpy(1.0, self.prior.mean)
                args[2].axpy(1.0, self.mean)
        else:
            raise NameError("Invalid number of parameters in Posterior::sample")

    def _sample_given_white_noise(
        self, noise: dlx.la.Vector, s_prior: dlx.la.Vector, s_post: dlx.la.Vector
    ):
        self.prior.sample(noise, s_prior, add_mean=False)
        self.sampler.sample(s_prior, s_post)

    def _sample_given_prior(self, s_prior: dlx.la.Vector, s_post: dlx.la.Vector):
        self.sampler.sample(s_prior, s_post)

    @not_implemented
    def trace(self, **kwargs):
        """
        Compute/estimate the trace of the posterior, prior distribution
        and the trace of the data informed correction.

        See :code:`_Prior.trace` for more details.
        """
        pr_trace = self.prior.trace(**kwargs)
        corr_trace = self.trace_update()
        post_trace = pr_trace - corr_trace
        return post_trace, pr_trace, corr_trace

    @not_implemented
    def trace_update(self):
        return self.Hlr.LowRankHinv.trace(self.prior.M)

    @not_implemented
    def pointwise_variance(self, **kwargs):
        """
        Compute/estimate the pointwise variance of the posterior, prior distribution
        and the pointwise variance reduction informed by the data.

        See :code:`_Prior.pointwise_variance` for more details.
        """
        pr_pointwise_variance = self.prior.pointwise_variance(**kwargs)
        # correction_pointwise_variance = Vector(self.prior.R.mpi_comm())
        correction_pointwise_variance = None
        self.init_vector(correction_pointwise_variance, 0)
        self.Hlr.LowRankHinv.get_diagonal(correction_pointwise_variance)
        post_pointwise_variance = pr_pointwise_variance - correction_pointwise_variance
        return (
            post_pointwise_variance,
            pr_pointwise_variance,
            correction_pointwise_variance,
        )

    def klDistanceFromPrior(self, sub_comp=False):
        dplus1 = self.d + np.ones_like(self.d)

        c_logdet = 0.5 * np.sum(np.log(dplus1))
        c_trace = -0.5 * np.sum(self.d / dplus1)
        c_shift = self.prior.cost(self.mean)

        kld = c_logdet + c_trace + c_shift

        if sub_comp:
            return kld, c_logdet, c_trace, c_shift
        else:
            return kld
