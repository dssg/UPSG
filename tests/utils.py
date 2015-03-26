import os, sys
import uuid
import unittest
import glob

import numpy as np

from upsg.utils import np_sa_to_nd

TESTS_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
DATA_PATH = os.path.join(TESTS_PATH, 'data')
REPO_PATH = os.path.join(TESTS_PATH, '..')
UPSG_PATH = os.path.join(REPO_PATH, 'upsg')
TEMP_PATH = os.path.join(TESTS_PATH, 'tmp')
BIN_PATH = os.path.join(REPO_PATH, 'bin')

def path_of_data(filename):
    return os.path.join(DATA_PATH, filename)

def csv_read(filename, as_nd = False):
    sa =  np.genfromtxt(filename, dtype=None, delimiter=",", names=True)
    if as_nd:
        return np_sa_to_nd(sa)[0]
    return sa


class TempFileManager:
    def __init__(self):
        if not os.path.exists(TEMP_PATH):
            os.makedirs(TEMP_PATH)
        self.__files = {}

    def get(self, filename = None):
        if filename is None:
            filename = uuid.uuid4()
        try:
            path = self.__files[filename]
        except KeyError:
            path = os.path.join(TEMP_PATH, filename)
            self.__files[filename] = path
        return path

    def __call__(self, filename = None):
        return self.get(filename)

    def purge(self):
        [os.remove(self.__files[filename]) for filename in self.__files
            if os.path.exists(self.__files[filename])]
        self.__files = {}

    def csv_read(self, filename, as_nd = False):
        return csv_read(self.__files[filename], as_nd)

class UPSGTestCase(unittest.TestCase):
    def setUp(self):
        self.__cwd = os.getcwd()
        if not os.path.exists(TEMP_PATH):
            os.makedirs(TEMP_PATH)
        os.chdir(TEMP_PATH)
        self._tmp_files = TempFileManager()
    def tearDown(self):
        os.system('{}/cleanup.py'.format(BIN_PATH))
        self._tmp_files.purge()
        os.chdir(self.__cwd)
