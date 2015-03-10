import numpy as np
from os import system
import unittest

from upsg.uobject import *
from utils import path_of_data

class TestUObject(unittest.TestCase):
    def test_csv_load_store(self):
        filename = path_of_data('mixed_csv.csv')
        uo = UObject(UObjectPhase.Write)
        uo.from_csv(filename)
        uo.write_to_read_phase()
        result = uo.to_np()
        control = np.genfromtxt(filename, dtype=None, delimiter=",", names=True)
        self.assertTrue(np.array_equal(result, control))
    def test_dict_load_store(self):
        d = {'k1' : 'A String that is fairly long', 'k2' : 42.1, 'k3' : 7}
        uo = UObject(UObjectPhase.Write)
        uo.from_dict(d)
        uo.write_to_read_phase()
        result = uo.to_dict()
        self.assertEqual(d, result)
    def tearDown(self):
        system('rm *.upsg')

if __name__ == '__main__':
    unittest.main()
