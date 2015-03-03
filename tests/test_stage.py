import numpy as np
import unittest
import importlib
import pickle
from os import system
from StringIO import StringIO

from utils import UPSG_PATH, path_of_data

from upsg.stage import Stage
from upsg.uobject import UObject, UObjectPhase

class TestStage(unittest.TestCase):

    def __pickle(self, import_path, *module_args, **module_kwargs):
        split_import_path = import_path.split('.')
        cls_name = split_import_path[-1]
        mod_name = '.'.join(split_import_path[:-1])
        mod = importlib.import_module(mod_name)
        stage = getattr(mod, cls_name)(*module_args, **module_kwargs)
        pickle_target = StringIO()
        pickle.dump(stage, pickle_target)
        pickle_target.seek(0)
        stage_recovered = pickle.load(pickle_target)
        self.assertEqual(stage.input_keys, stage_recovered.input_keys)
        self.assertEqual(stage.output_keys, stage_recovered.output_keys)

    def test_pickle(self):
    # TODO this just makes sure the object can be pickled. It doesn't verify
    #   that the unpickled object is correct
    # TODO wrapped methods
        uo = UObject(UObjectPhase.Write)
        np_array = np.array([[0]])
        uo.from_np(np_array)
        self.__pickle('upsg.export.csv.CSVWrite',  path_of_data('_out.csv'))
        self.__pickle('upsg.fetch.csv.CSVRead',  path_of_data('mixed_csv.csv'))
        self.__pickle('upsg.fetch.np.NumpyRead',  np.array([[0]]))
        self.__pickle('upsg.transform.split.SplitTrainTest')
        self.__pickle('upsg.transform.split.SplitColumn', 0)

    def tearDown(self):
        system('rm *.upsg')

if __name__ == '__main__':
    unittest.main()

#'upsg.fetch.uobject.UObjectRead': <class 'upsg.fetch.uobject.UObjectRead'>, 'upsg.transform.split.SplitTrainTest': <class 'upsg.transform.split.SplitTrainTest'>, 'upsg.fetch.csv.CSVRead': <class 'upsg.fetch.csv.CSVRead'>, 'upsg.transform.split.SplitColumn': <class 'upsg.transform.split.SplitColumn'>, 'upsg.export.csv.CSVWrite': <class 'upsg.export.csv.CSVWrite'>, 'upsg.fetch.np.NumpyRead': <class 'upsg.fetch.np.NumpyRead'>}
