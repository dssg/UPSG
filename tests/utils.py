import os
import sys
import uuid
import unittest
import glob
import shutil
import gc
from HTMLParser import HTMLParser

import numpy as np

import upsg
from upsg.utils import np_sa_to_nd

TESTS_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
DATA_PATH = os.path.join(TESTS_PATH, 'data')
UPSG_PATH = os.path.dirname(upsg.__file__)
TEMP_PATH = os.path.join(TESTS_PATH, 'tmp')
BIN_PATH = os.path.join(UPSG_PATH, 'bin')


def path_of_data(filename):
    return os.path.join(DATA_PATH, filename)


def csv_read(filename, as_nd=False, dtype=None):
    sa = np.genfromtxt(filename, dtype=dtype, delimiter=",", names=True)
    if as_nd:
        return np_sa_to_nd(sa)[0]
    return sa


class TempFileManager(object):

    def __init__(self):
        if not os.path.exists(TEMP_PATH):
            os.makedirs(TEMP_PATH)
        self.__files = {}

    def get(self, filename):
        try:
            path = self.__files[filename]
        except KeyError:
            path = os.path.join(TEMP_PATH, filename)
            self.__files[filename] = path
        return path

    def __call__(self, filename=None):
        return self.get(filename)

    class __PurgeFromHTML(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == 'img':
                for name, value in attrs:
                    if name == 'src':
                        os.remove(value)
    
    __purge_from_HTML_inst = __PurgeFromHTML()

    def __purge_file(self, filename):
        path = self.__files[filename]
        if not os.path.exists(path):
            return
        if path.split('.')[-1] == 'html':
            with open(path) as fin:
                self.__purge_from_HTML_inst.feed(fin.read())
        os.remove(path)

    def purge(self):
        [self.__purge_file(filename) for filename in self.__files]
        self.__files = {}

    def csv_read(self, filename, as_nd=False):
        return csv_read(self.get(filename), as_nd)

    def tmp_copy(self, from_path):
        ext = os.path.splitext(from_path)[1]
        filename = str(uuid.uuid4()) + ext
        to_path = self.get(filename)
        shutil.copyfile(from_path, to_path)
        return (to_path, filename)

class UPSGTestCase(unittest.TestCase):

    def setUp(self):
        self.__cwd = os.getcwd()
        if not os.path.exists(TEMP_PATH):
            os.makedirs(TEMP_PATH)
        os.chdir(TEMP_PATH)
        self._tmp_files = TempFileManager()

    def tearDown(self):
        gc.collect()
        os.system('python {}'.format(os.path.join(BIN_PATH, 'cleanup.py')))
        self._tmp_files.purge()
        os.chdir(self.__cwd)
