import numpy as np
from os import system
import unittest

from upsg.uobject import *
from upsg.utils import np_type
from utils import path_of_data

class TestUObject(unittest.TestCase):
    test_dict = {'k1' : 'A String that is fairly long', 'k2' : 42.1, 'k3' : 7}
    test_dict_keys = test_dict.keys()
    test_array_dtype = np.dtype({'names' : test_dict_keys, 'formats' : 
        [np_type(test_dict[key]) for key in test_dict_keys]})
    test_array_vals = [tuple([test_dict[key] for key in test_dict_keys])]
    test_array = np.array(test_array_vals, dtype = test_array_dtype)
    def test_csv_load_store(self):
        filename = path_of_data('mixed_csv.csv')
        uo = UObject(UObjectPhase.Write)
        uo.from_csv(filename)
        uo.write_to_read_phase()
        result = uo.to_np()
        control = np.genfromtxt(filename, dtype=None, delimiter=",", names=True)
        self.assertTrue(np.array_equal(result, control))
    def test_dict_load_store(self):
        d = self.test_dict
        uo = UObject(UObjectPhase.Write)
        uo.from_dict(d)
        uo.write_to_read_phase()
        result = uo.to_dict()
        self.assertEqual(d, result)
    def test_dict_to_np(self):
        uo = UObject(UObjectPhase.Write)
        uo.from_dict(self.test_dict)
        uo.write_to_read_phase()
        A = uo.to_np()
        self.assertTrue(np.array_equal(self.test_array, A))
        self.assertEqual(self.test_array.dtype, A.dtype)
    def test_np_to_dict(self):
        uo = UObject(UObjectPhase.Write)
        uo.from_np(self.test_array)
        uo.write_to_read_phase()
        d = uo.to_dict()
        self.assertEqual(d, self.test_dict)
    def tearDown(self):
        system('rm *.upsg')

if __name__ == '__main__':
    unittest.main()
