import os, sys

TESTS_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
DATA_PATH = os.path.join(TESTS_PATH, 'data')
UPSG_PATH = os.path.join(TESTS_PATH, '..', 'upsg')

def path_of_data(filename):
    return os.path.join(DATA_PATH, filename)


    
