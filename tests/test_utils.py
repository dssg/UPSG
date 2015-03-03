import numpy as np
from os import system
import unittest

from upsg.utils import *

class TestUObject(unittest.TestCase):
    def test_nd_to_sa_w_type(self):
        nd = np.array([[1,2,3],[4,5,6]], dtype = int)
        dtype = np.dtype({'names' : map('f{}'.format, xrange(3)), 
            'formats' : [int] * 3})
        control = np.array([(1,2,3),(4,5,6)], dtype=dtype) 
        result = np_nd_to_sa(nd, dtype)
        self.assertEqual(control.dtype, result.dtype)
        self.assertTrue(np.array_equal(result, control))
    def test_nd_to_sa_no_type(self):
        nd = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]], dtype = float)
        dtype = np.dtype({'names' : map('f{}'.format, xrange(3)), 
            'formats' : [float] * 3})
        control = np.array([(1.0,2.0,3.0),(4.0,5.0,6.0)], dtype=dtype) 
        result = np_nd_to_sa(nd)
        self.assertEqual(control.dtype, result.dtype)
        self.assertTrue(np.array_equal(result, control))
    def test_sa_to_nd(self):
        dtype = np.dtype({'names' : map('f{}'.format, xrange(3)), 
            'formats' : [float] * 3})
        sa = np.array([(-1.0, 2.0, -1.0), (0.0, -1.0, 2.0)], dtype = dtype)
        control = np.array([[-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]], 
            dtype = float)
        (result, sa_dtype) = np_sa_to_nd(sa)
        self.assertEqual(dtype, sa_dtype)
        self.assertEqual(control.dtype, result.dtype)
        self.assertTrue(np.array_equal(result, control))
    def test_is_sa(self):
        nd = np.array([[1,2,3],[4,5,6]], dtype = int)
        dtype = np.dtype({'names' : map('f{}'.format, xrange(3)), 
            'formats' : [float] * 3})
        sa = np.array([(-1.0, 2.0, -1.0), (0.0, -1.0, 2.0)], dtype = dtype)
        self.assertFalse(is_sa(nd))
        self.assertTrue(is_sa(sa))
            
if __name__ == '__main__':
    unittest.main()
