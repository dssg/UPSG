import os, sys
import uuid

import numpy as np

from upsg.utils import np_sa_to_nd

TESTS_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
DATA_PATH = os.path.join(TESTS_PATH, 'data')
UPSG_PATH = os.path.join(TESTS_PATH, '..', 'upsg')
TEMP_PATH = os.path.join(TESTS_PATH, 'tmp')

def path_of_data(filename):
    return os.path.join(DATA_PATH, filename)

def csv_read(filename, as_nd = False):
    sa =  np.genfromtxt(filename, dtype=None, delimiter=",", names=True)
    if as_nd:
        return np_sa_to_nd(sa)
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
            path = self.__files[key]
        except KeyError:
            path = os.path.join(TEMP_PATH, filename)
            self.__files[filename] = path
        return path
    def purge(self):
        [os.remove(self.__files[filename]) for filename in self.__files
            if os.path.exists(self.__files[filename])]
        self.__files = {}
