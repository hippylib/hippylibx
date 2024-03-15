# import unittest
# import subprocess
# import sys
# import os

# sys.path.append(os.path.abspath('../..'))

# import hippylibX as hpx
# sys.path.append(os.path.abspath('../../example'))

# from poisson_example import run_inversion

# class TestFile1Execution(unittest.TestCase):
#     os.chdir("../../example/")
#     def test_execution_poisson_single(self):
#         command = "mpirun -n 1 python3 -u poisson_example.py"
#         return_code = os.system(command)
#         self.assertEqual(return_code, 0, "Error running 1 proc poisson_example.py")

# if __name__ == "__main__":
#     unittest.main()

###################################################################


# import unittest
# from unittest.mock import patch
# import sys
# import os

# sys.path.append(os.path.abspath('../..'))

# import hippylibX
# from hippylibX import modelVerify

# sys.path.append(os.path.abspath('../../example'))

# from poisson_example import run_inversion



# class TestFileExecution(unittest.TestCase):
#     @patch('hippylibX.modelVerify')
#     def test_verify_function_called(self, mock_verify):
#         # Mock the verify function
#         mock_verify.return_value = (1, 2, 3, 4)
        
#         nx = 64
#         ny = 64
#         noise_variance = 1e-4
#         prior_param = {"gamma": 0.1, "delta": 1.}
#         result = run_inversion(nx, ny, noise_variance, prior_param)

#         # Verify that the verify function was called
#         mock_verify.assert_called_once()

#         # Perform assertions on the returned values
#         print(result)
#         # var1, var2, var3, var4 = result
#         # self.assertTrue(var1 + var2 == 3, "var1 + var2 should be equal to 3")
#         # self.assertTrue(var3 * var4 == 12, "var3 * var4 should be equal to 12")

# if __name__ == "__main__":
#     unittest.main()


###################################################################

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
        command = "mpirun -n 4 python3 -u sfsi_toy_gaussian.py"
        return_val = os.system(command)
        os.chdir(pwd)
        self.assertEqual(return_val, 0, "4 proc qpact: Error running")

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
        command = "mpirun -n 4 python3 -u poisson_example.py"
        return_val = os.system(command)
        os.chdir(pwd)
        self.assertEqual(return_val, 0, "4 proc poisson: Error running")

    
    # def test_qpact_results_serial_misfit_True(self):

    #     with open('outputs_qpact_1_proc_misfit_True.pickle', 'rb') as f:
    #         data = pickle.load(f)

    #     sym_Hessian_value, slope_grad, slope_H = data_parser(data)

    #     self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "1 proc qpact misfit True: Symmetric Hessian check value is greater than 1e-10")
    #     self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="1 proc qpact misfit True: FD Gradient check slope is not close to 1")
    #     self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="1 proc qpact misfit True: FD Hessian check slope is not close to 1")

    # def test_qpact_results_serial_misfit_False(self):
    #     #misfit=False
    #     with open('outputs_qpact_1_proc_misfit_False.pickle', 'rb') as f:
    #         data = pickle.load(f)
    #     sym_Hessian_value, slope_grad, slope_H = data_parser(data)
    #     self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "1 proc qpact misfit False: Symmetric Hessian check value is greater than 1e-10")
    #     self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="1 proc qpact misfit False: FD Gradient check slope is not close to 1")
    #     self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="1 proc qpact misfit False: FD Hessian check slope is not close to 1")

    # def test_qpact_results_parallel_misfit_True(self):

    #     with open('outputs_qpact_4_proc_misfit_True.pickle', 'rb') as f:
    #         data = pickle.load(f)

    #     sym_Hessian_value, slope_grad, slope_H = data_parser(data)

    #     self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "4 proc qpact misfit True: Symmetric Hessian check value is greater than 1e-10")
    #     self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="4 proc qpact misfit True: FD Gradient check slope is not close to 1")
    #     self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="4 proc qpact misfit True: FD Hessian check slope is not close to 1")

    # def test_qpact_results_parallel_misfit_False(self):
    #     #misfit=False
    #     with open('outputs_qpact_4_proc_misfit_False.pickle', 'rb') as f:
    #         data = pickle.load(f)
    #     sym_Hessian_value, slope_grad, slope_H = data_parser(data)
    #     self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "4 proc qpact misfit False: Symmetric Hessian check value is greater than 1e-10")
    #     self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="4 proc qpact misfit False: FD Gradient check slope is not close to 1")
    #     self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="4 proc qpact misfit False: FD Hessian check slope is not close to 1")


    # def test_poisson_results_serial_misfit_True(self):

    #     with open('outputs_poisson_1_proc_misfit_True.pickle', 'rb') as f:
    #         data = pickle.load(f)

    #     sym_Hessian_value, slope_grad, slope_H = data_parser(data)

    #     self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "1 proc poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
    #     self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="1 proc poisson misfit True: FD Gradient check slope is not close to 1")
    #     self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="1 proc poisson misfit True: FD Hessian check slope is not close to 1")

    # def test_poisson_results_serial_misfit_False(self):
    #     #misfit=False
    #     with open('outputs_poisson_1_proc_misfit_False.pickle', 'rb') as f:
    #         data = pickle.load(f)
    #     sym_Hessian_value, slope_grad, slope_H = data_parser(data)
    #     self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "1 proc poisson misfit False: Symmetric Hessian check value is greater than 1e-10")
    #     self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="1 proc poisson misfit False: FD Gradient check slope is not close to 1")
    #     self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="1 proc poisson misfit False: FD Hessian check slope is not close to 1")

    # def test_poisson_results_parallel_misfit_True(self):
    #     with open('outputs_poisson_4_proc_misfit_True.pickle', 'rb') as f:
    #         data = pickle.load(f)

    #     sym_Hessian_value, slope_grad, slope_H = data_parser(data)

    #     self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "4 proc poisson misfit True: Symmetric Hessian check value is greater than 1e-10")
    #     self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="4 proc poisson misfit True: FD Gradient check slope is not close to 1")
    #     self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="4 proc poisson misfit True: FD Hessian check slope is not close to 1")

    # def test_qpact_results_parallel_misfit_False(self):
    #     #misfit=False
    #     with open('outputs_poisson_4_proc_misfit_False.pickle', 'rb') as f:
    #         data = pickle.load(f)
    #     sym_Hessian_value, slope_grad, slope_H = data_parser(data)
    #     self.assertLessEqual(np.abs(sym_Hessian_value), 1e-10, "4 proc poisson misfit False: Symmetric Hessian check value is greater than 1e-10")
    #     self.assertAlmostEqual(slope_grad, 1, delta=1e-1, msg="4 proc poisson misfit False: FD Gradient check slope is not close to 1")
    #     self.assertAlmostEqual(slope_H, 1, delta=1e-1, msg="4 proc poisson misfit False: FD Hessian check slope is not close to 1")


        
if __name__ == "__main__":
    unittest.main()






