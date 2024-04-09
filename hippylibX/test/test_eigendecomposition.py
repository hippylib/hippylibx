import petsc4py
import sys
import os

sys.path.append(os.path.abspath("../.."))
import hippylibX as hpx
import numpy as np
import unittest

sys.path.append(os.path.abspath("../../example"))

from example import poisson_dirichlet_example


def check_g(
    A: petsc4py.PETSc.Mat, B: petsc4py.PETSc.Mat, U: hpx.MultiVector, d: np.array
) -> tuple[float, float, np.array]:
    nvec = U.nvec
    AU = hpx.MultiVector.createFromVec(U[0], nvec)
    BU = hpx.MultiVector.createFromVec(U[0], nvec)
    hpx.MatMvMult(A, U, AU)
    hpx.MatMvMult(B, U, BU)

    # Residual checks
    diff = hpx.MultiVector.createFromMultiVec(AU)
    diff.axpy(-d, BU)

    # res_norms = diff.norm("l2")
    res_norms = diff.norm(petsc4py.PETSc.NormType.N2)

    # B-ortho check
    UtBU = BU.dot(U)

    err = UtBU - np.eye(nvec, dtype=UtBU.dtype)
    err_Bortho = np.linalg.norm(err, "fro")

    # A-ortho check
    V = hpx.MultiVector.createFromMultiVec(U)
    scaling = np.power(np.abs(d), -0.5)
    V.scale(scaling)
    AU.scale(scaling)
    VtAV = AU.dot(V)
    err = VtAV - np.eye(nvec, dtype=VtAV.dtype)
    err_Aortho = np.linalg.norm(err, "fro")

    return err_Bortho, err_Aortho, res_norms


def check_output(self, result: tuple[float, float, np.array]):
    self.assertLessEqual(
        np.abs(result[0]),
        1e-10,
        "Frobenius norm of U^TBU - I_k not less than 1e-10",
    )

    self.assertLessEqual(
        np.abs(result[1]),
        1e-10,
        "Frobenius norm of V^TAV) - I_k not less than 1e-10",
    )

    for i in range(len(result[2])):
        self.assertLessEqual(
            np.abs(result[2][i]),
            1e-1,
            f"l2 norm of residual r[{i}] = A U[{i}] - d[{i}] B U[{i}] not less than 1e-1",
        )


class Testing_Execution(unittest.TestCase):
    def test_eigendecomposition(self):
        hpx.parRandom.replay()
        nx = 64
        ny = 64
        noise_variance = 1e-4
        prior_param = {"gamma": 0.1, "delta": 1.0}
        out = poisson_dirichlet_example.run_inversion(
            nx, ny, noise_variance, prior_param
        )
        A, B, d, U = (
            out["eigen_decomposition_results"]["A"],
            out["eigen_decomposition_results"]["B"],
            out["eigen_decomposition_results"]["d"],
            out["eigen_decomposition_results"]["U"],
        )
        result = check_g(A, B, U, d)
        check_output(self, result)


if __name__ == "__main__":
    unittest.main()
