import unittest
import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../.."))

import hippylibX as hpx

sys.path.append(os.path.abspath("../misc"))

from misc import multivector_sample_script

def check_output(self,result):
    self.assertTrue(result.shape[0] == result.shape[1])
    self.assertTrue(np.allclose(result, np.eye(result.shape[0]),atol=1e-6))


class Testing_Execution(unittest.TestCase):
    def test_qpact_bilap_prior_execution(self):
        hpx.parRandom.replay()
        nx = 64
        ny = 64
        nvec = 10
        result = multivector_sample_script.multi_vector_testing(nx, ny, nvec)
        check_output(self,result)

if __name__ == "__main__":
    unittest.main()
