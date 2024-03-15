
import unittest
import sys
import os
import numpy as np
import pickle


sys.path.append(os.path.abspath('../..'))

import hippylibX as hpx
sys.path.append(os.path.abspath('../../example'))


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


class Testing_Execution(unittest.TestCase):

    def test_poisson_execution_serial(self):
        pwd = os.getcwd()
        os.chdir("../../example/")
        command = "mpirun -n 1 python3 -u poisson_example.py"
        return_val = os.system(command)
        os.chdir(pwd)
        self.assertEqual(return_val, 0, "1 proc poisson: Error running")

    def test_poisson_execution_parallel(self):
        pwd = os.getcwd()
        os.chdir("../../example/")
        command = "mpirun -n 1 python3 -u poisson_example.py"
        return_val = os.system(command)
        os.chdir(pwd)
        self.assertEqual(return_val, 0, "4 proc poisson: Error running")

    def test_qpact_execution_serial(self):
        pwd = os.getcwd()
        os.chdir("../../example/")
        command = "mpirun -n 1 python3 -u sfsi_toy_gaussian.py"
        return_val = os.system(command)
        os.chdir(pwd)
        self.assertEqual(return_val, 0, "1 proc qpact: Error running")

    def test_qpact_execution_parallel(self):
        pwd = os.getcwd()
        os.chdir("../../example/")
        command = "mpirun -n 1 python3 -u sfsi_toy_gaussian.py"
        return_val = os.system(command)
        os.chdir(pwd)
        self.assertEqual(return_val, 0, "4 proc qpact: Error running")

    def test_qpact_results_serial_misfit_True(self):

        with open('outputs_qpact_1_proc_misfit_True.pickle', 'rb') as f:
            data = pickle.load(f)

        sym_Hessian_value, slope_grad, slope_H = data_parser(data)

        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "1 proc qpact misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="1 proc qpact misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="1 proc qpact misfit True: FD Hessian check slope is not close to 1")

    def test_qpact_results_serial_misfit_False(self):
        #misfit=False
        with open('outputs_qpact_1_proc_misfit_False.pickle', 'rb') as f:
            data = pickle.load(f)
        sym_Hessian_value, slope_grad, slope_H = data_parser(data)
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "1 proc qpact misfit False: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="1 proc qpact misfit False: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="1 proc qpact misfit False: FD Hessian check slope is not close to 1")

    def test_qpact_results_parallel_misfit_True(self):

        with open('outputs_qpact_4_proc_misfit_True.pickle', 'rb') as f:
            data = pickle.load(f)

        sym_Hessian_value, slope_grad, slope_H = data_parser(data)

        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "4 proc qpact misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="4 proc qpact misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="4 proc qpact misfit True: FD Hessian check slope is not close to 1")

    def test_qpact_results_parallel_misfit_False(self):
        #misfit=False
        with open('outputs_qpact_4_proc_misfit_False.pickle', 'rb') as f:
            data = pickle.load(f)
        sym_Hessian_value, slope_grad, slope_H = data_parser(data)
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "4 proc qpact misfit False: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="4 proc qpact misfit False: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="4 proc qpact misfit False: FD Hessian check slope is not close to 1")


    def test_poisson_results_serial_misfit_True(self):

        with open('outputs_poisson_1_proc_misfit_True.pickle', 'rb') as f:
            data = pickle.load(f)

        sym_Hessian_value, slope_grad, slope_H = data_parser(data)

        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "1 proc poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="1 proc poisson misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="1 proc poisson misfit True: FD Hessian check slope is not close to 1")

    def test_poisson_results_serial_misfit_False(self):
        #misfit=False
        with open('outputs_poisson_1_proc_misfit_False.pickle', 'rb') as f:
            data = pickle.load(f)
        sym_Hessian_value, slope_grad, slope_H = data_parser(data)
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "1 proc poisson misfit False: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="1 proc poisson misfit False: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="1 proc poisson misfit False: FD Hessian check slope is not close to 1")

    def test_poisson_results_parallel_misfit_True(self):
        with open('outputs_poisson_4_proc_misfit_True.pickle', 'rb') as f:
            data = pickle.load(f)

        sym_Hessian_value, slope_grad, slope_H = data_parser(data)

        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "4 proc poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="4 proc poisson misfit True: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="4 proc poisson misfit True: FD Hessian check slope is not close to 1")

    def test_qpact_results_parallel_misfit_False(self):
        #misfit=False
        with open('outputs_poisson_4_proc_misfit_False.pickle', 'rb') as f:
            data = pickle.load(f)
        sym_Hessian_value, slope_grad, slope_H = data_parser(data)
        self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "4 proc poisson misfit False: Symmetric Hessian check value is greater than 1e-10")
        self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="4 proc poisson misfit False: FD Gradient check slope is not close to 1")
        self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="4 proc poisson misfit False: FD Hessian check slope is not close to 1")


        
if __name__ == "__main__":
    unittest.main()






