import unittest

suite = unittest.defaultTestLoader.discover('.')

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite)
