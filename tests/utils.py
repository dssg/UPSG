import os, sys

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),
    'data')

def path_of_data(filename):
    return os.path.join(DATA_PATH, filename)


    
