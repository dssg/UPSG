import unittest
import utils

suite = unittest.defaultTestLoader.discover(utils.TESTS_PATH)

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite)
