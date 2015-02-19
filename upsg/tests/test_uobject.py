from upsg.uobject import *
import numpy as np
from os import system
import unittest

class TestUObject(unittest.TestCase):
    def test_csv_load_store(self):
        filename = 'mixed_csv.csv'
        uo = UObject(UObjectPhase.Write)
        uo.from_csv(filename)
        uo.to_read_phase()
        result = uo.to_np()
        control = np.genfromtxt(filename, dtype=None, delimiter=",", names=True)
        self.assertTrue(np.array_equal(result, control))
    def tearDown(self):
        system('rm *.upsg')

if __name__ == '__main__':
    unittest.main()
