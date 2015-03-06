import numpy as np
from os import system
import unittest
import inspect
from StringIO import StringIO

from upsg.pipeline import Pipeline
from upsg.export.csv import CSVWrite
from upsg.fetch.csv import CSVRead
from upsg.wrap.wrap_sklearn import wrap_instance
from upsg.stage import Stage
from upsg.uobject import UObject, UObjectPhase
from utils import path_of_data
from upsg.utils import np_nd_to_sa, np_sa_to_nd

outfile_name = path_of_data('_out.csv')

class LambdaStage(Stage):
    def __init__(self, lam, fout = None):
        self.__lam = lam
        self.__input_keys = dict.fromkeys(inspect.getargspec(lam).args, True)
        self.__fout = fout
        if fout:
            self.__output_keys = []
        else:
            self.__output_keys = ['fx']

    @property
    def input_keys(self):
        return self.__input_keys

    @property
    def output_keys(self):
        return self.__output_keys

    def run(self, outputs_requested, **kwargs):
        fx = self.__lam(**{key : kwargs[key].to_np()[0][0] 
            for key in kwargs}) 
        if self.__fout:
            self.__fout.write(str(fx))
            return {}
        fx_np = np.core.records.fromrecords([(fx,)])
        uo = UObject(UObjectPhase.Write)
        uo.from_np(fx_np)
        return {'fx': uo}

class TestPipleline(unittest.TestCase):
    def test_rw(self):
        infile_name = path_of_data('mixed_csv.csv')        

        p = Pipeline()

        csv_read_node = p.add(CSVRead(infile_name))
        csv_write_node = p.add(CSVWrite(outfile_name))

        csv_read_node['out'] > csv_write_node['in']

        p.run()

        control = np.genfromtxt(infile_name, dtype=None, delimiter=",", 
            names=True)
        result = np.genfromtxt(outfile_name, dtype=None, delimiter=",", 
            names=True)
        
        self.assertTrue(np.array_equal(result, control))

    def test_3_stage(self):
        from sklearn.preprocessing import Imputer
    
        infile_name = path_of_data('missing_vals.csv')        

        p = Pipeline()

        csv_read_node = p.add(CSVRead(infile_name))
        csv_write_node = p.add(CSVWrite(outfile_name))
        impute_node = p.add(wrap_instance(Imputer))

        csv_read_node['out'] > impute_node['X_train']
        impute_node['X_new'] > csv_write_node['in']

        p.run()

        ctrl_imputer = Imputer()
        ctrl_X_sa = np.genfromtxt(infile_name, dtype=None, delimiter=",", 
            names=True)
        num_type = ctrl_X_sa[0][0].dtype
        ctrl_X_nd, ctrl_X_sa_type = np_sa_to_nd(ctrl_X_sa)
        ctrl_X_new_nd = ctrl_imputer.fit_transform(ctrl_X_nd)
        control = ctrl_X_new_nd

        res_sa = np.genfromtxt(outfile_name, dtype=None, delimiter=",", 
            names=True)
        result, res_sa_dtype = np_sa_to_nd(res_sa)
        
        self.assertTrue(np.allclose(result, control))

    def test_DAG(self):
        p = Pipeline()

        s0 = LambdaStage(lambda: 'S0')
        s1 = LambdaStage(lambda: 'S1')
        s2 = LambdaStage(lambda: 'S2')
        s3 = LambdaStage(lambda x, y: '({},{})->I{}'.format(x, y, '3'))
        s4 = LambdaStage(lambda x, y: '({},{})->I{}'.format(x, y, '4'))
        s5out = StringIO()
        s6out = StringIO()
        s5 = LambdaStage(lambda x, y: '({},{})->T{}'.format(x, y, '5'), 
            fout = s5out)
        s6 = LambdaStage(lambda x: '({})->T{}'.format(x, '6'), 
            fout = s6out)
        nodes = [p.add(s) for s in (s0, s1, s2, s3, s4, s5, s6)]

        nodes[0]['fx'] > nodes[3]['x']
        nodes[1]['fx'] > nodes[3]['y']
        nodes[1]['fx'] > nodes[4]['x']
        nodes[2]['fx'] > nodes[4]['y']
        nodes[3]['fx'] > nodes[5]['x']
        nodes[4]['fx'] > nodes[5]['y']
        nodes[4]['fx'] > nodes[6]['x']

        p.run()

        self.assertEqual(s5out.getvalue(), 
            "((S0,S1)->I3,(S1,S2)->I4)->T5")
        self.assertEqual(s6out.getvalue(),
            "((S1,S2)->I4)->T6")

    def tearDown(self):
        system('rm *.upsg')
        system('rm {}'.format(outfile_name))

if __name__ == '__main__':
    unittest.main()
