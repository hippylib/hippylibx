import unittest
import sys
import os
import numpy as np
from mpi4py import MPI


sys.path.append(os.path.abspath('../..'))

import hippylibX as hpx
sys.path.append(os.path.abspath('../../example'))


from example import poisson_example, sfsi_toy_gaussian, poisson_dirichlet_example
from example import poisson_example_reg, sfsi_toy_gaussian_reg, poisson_dirichlet_example_reg

def data_parser(data):
        eps = data["eps"]
        err_grad = data['err_grad']
        err_H = data['err_H']
        sym_Hessian_value = data['sym_Hessian_value']

        slope_grad_coeffs = np.polyfit(np.log(eps[20:30]), np.log(err_grad[20:30]), 1)
        slope_grad = slope_grad_coeffs[0]

        slope_H_coeffs = np.polyfit(np.log(eps[20:30]), np.log(err_H[20:30]), 1)
        slope_H = slope_H_coeffs[0]

        return sym_Hessian_value, slope_grad, slope_H


class Test_runner:
    def __init__(self):
        self.result = unittest.TestResult()

    def run_tests(self):
        test_suite = unittest.TestLoader().loadTestsFromTestCase(Testing_Execution)
        test_suite.run(self.result)

        return 1 if self.result.wasSuccessful() else 0  
    

class Testing_Execution(unittest.TestCase):

    def test_qpact_bilap_prior_execution(self):
        hpx.parRandom.replay()
        nx = 64 
        ny = 64
        noise_variance = 1e-6
        prior_param = {"gamma": 0.1, "delta": 2.}
        mesh_filename = '../../example/meshes/circle.xdmf'
        out = sfsi_toy_gaussian.run_inversion(mesh_filename, nx, ny, noise_variance, prior_param)

        #convergence of optimizer
        self.assertEqual(out['optimizer_results']['optimizer'],True,"Did not converge")
        
        # misfit = True, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_True'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "qpact misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="qpact misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="qpact misfit True: FD Hessian check slope is not close to 1")

        # misfit = False, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_False'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "qpact misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="qpact misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="qpact misfit True: FD Hessian check slope is not close to 1")

    def test_poisson_robin_bilap_prior_execution(self):
        hpx.parRandom.replay()
        nx = 64 
        ny = 64
        noise_variance = 1e-4
        prior_param = {"gamma": 0.1, "delta": 1.}
        out = poisson_example.run_inversion(nx, ny, noise_variance, prior_param)
        
        #convergence of optimizer
        self.assertEqual(out['optimizer_results']['optimizer'],True,"Did not converge")
        
        # misfit = True, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_True'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="poisson misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="poisson misfit True: FD Hessian check slope is not close to 1")

        # misfit = False, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_False'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="poisson misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="poisson misfit True: FD Hessian check slope is not close to 1")

    def test_poisson_dirichlet_bilap_prior_execution(self):
        hpx.parRandom.replay()
        nx = 64 
        ny = 64
        noise_variance = 1e-4
        prior_param = {"gamma": 0.1, "delta": 1.}
        out = poisson_dirichlet_example.run_inversion(nx, ny, noise_variance, prior_param)
        
        #convergence of optimizer
        self.assertEqual(out['optimizer_results']['optimizer'],True,"Did not converge")
        
        # misfit = True, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_True'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="poisson misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="poisson misfit True: FD Hessian check slope is not close to 1")

        # misfit = False, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_False'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="poisson misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="poisson misfit True: FD Hessian check slope is not close to 1")

    def test_qpact_var_reg_prior_execution(self):
        hpx.parRandom.replay()
        nx = 64 
        ny = 64
        noise_variance = 1e-6
        prior_param = {"gamma": 0.1, "delta": 2.}
        mesh_filename = '../../example/meshes/circle.xdmf'


        out = sfsi_toy_gaussian_reg.run_inversion(mesh_filename, nx, ny, noise_variance, prior_param)

        #convergence of optimizer
        self.assertEqual(out['optimizer_results']['optimizer'],True,"Did not converge")
        
        # misfit = True, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_True'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "qpact misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="qpact misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="qpact misfit True: FD Hessian check slope is not close to 1")

        # misfit = False, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_False'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "qpact misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="qpact misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="qpact misfit True: FD Hessian check slope is not close to 1")

    def test_poisson_var_reg_prior_execution(self):
        hpx.parRandom.replay()
        nx = 64 
        ny = 64
        noise_variance = 1e-4
        prior_param = {"gamma": 0.1, "delta": 1.}
        out = poisson_example_reg.run_inversion(nx, ny, noise_variance, prior_param)
        
        #convergence of optimizer
        self.assertEqual(out['optimizer_results']['optimizer'],True,"Did not converge")
        
        # misfit = True, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_True'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="poisson misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="poisson misfit True: FD Hessian check slope is not close to 1")

        # misfit = False, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_False'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="poisson misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="poisson misfit True: FD Hessian check slope is not close to 1")

    def test_poisson_dirichlet_var_reg_prior_execution(self):
        hpx.parRandom.replay()
        nx = 64 
        ny = 64
        noise_variance = 1e-4
        prior_param = {"gamma": 0.1, "delta": 1.}
        out = poisson_dirichlet_example_reg.run_inversion(nx, ny, noise_variance, prior_param)
        
        #convergence of optimizer
        self.assertEqual(out['optimizer_results']['optimizer'],True,"Did not converge")
        
        # misfit = True, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_True'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="poisson misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="poisson misfit True: FD Hessian check slope is not close to 1")

        # misfit = False, slope and symmmetric nature of Hessian
        sym_Hessian_value, slope_grad, slope_H = data_parser(out['data_misfit_False'])
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="poisson misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="poisson misfit True: FD Hessian check slope is not close to 1")
        
if __name__ == "__main__":
    test_suite = unittest.defaultTestLoader.discover('.','testing_suite_file.py')
    test_runner = unittest.TextTestRunner(resultclass=unittest.TextTestResult)
    result = test_runner.run(test_suite)
    sys.exit(not result.wasSuccessful())















