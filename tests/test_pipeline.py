import numpy as np
from os import system
import unittest
import inspect
from StringIO import StringIO

from upsg.pipeline import Pipeline
from upsg.export.csv import CSVWrite
from upsg.fetch.csv import CSVRead
from upsg.wrap.wrap_sklearn import wrap_and_make_instance
from upsg.stage import RunnableStage
from upsg.uobject import UObject, UObjectPhase
from utils import path_of_data, UPSGTestCase
from upsg.utils import np_nd_to_sa, np_sa_to_nd


class OneCellLambdaStage(RunnableStage):

    def __init__(self, lam, fout=None, n_results=1):
        self.__lam = lam
        self.__input_keys = inspect.getargspec(lam).args
        self.__fout = fout
        self.__n_results = n_results
        if fout:
            self.__output_keys = []
        else:
            if n_results > 1:
                self.__output_keys = ['fx{}'.format(i)
                                      for i in xrange(n_results)]
            else:
                self.__output_keys = ['fx']

    @property
    def input_keys(self):
        return self.__input_keys

    @property
    def output_keys(self):
        return self.__output_keys

    def run(self, outputs_requested, **kwargs):
        fxs = self.__lam(**{key: kwargs[key].to_np()[0][0]
                            for key in kwargs})
        if self.__fout:
            self.__fout.write(str(fxs))
            return {}
        if self.__n_results <= 1:
            fxs = [fxs]
        fxs_np = map(lambda fx: np.core.records.fromrecords([(fx,)]),
                     fxs)
        ret = {key: UObject(UObjectPhase.Write) for key in self.__output_keys}
        [ret[key].from_np(fxs_np[i])
            for i, key in enumerate(self.__output_keys)]
        return ret

class MockupStage(RunnableStage):
    def __init__(self, in_keys, out_keys):
        self.__in_keys = in_keys
        self.__out_keys = out_keys

    @property
    def input_keys(self):
        return self.__in_keys

    @property
    def output_keys(self):
        return self.__out_keys

    def run(self, outputs_requested, **kwargs):
        return {}

class TestPipeline(UPSGTestCase):

    def test_rw(self):
        infile_name = path_of_data('mixed_csv.csv')

        p = Pipeline()

        csv_read_node = p.add(CSVRead(infile_name))
        csv_write_node = p.add(CSVWrite(self._tmp_files.get('out.csv')))

        csv_read_node['output'] > csv_write_node['input']

        p.run()

        control = np.genfromtxt(infile_name, dtype=None, delimiter=",",
                                names=True)
        result = self._tmp_files.csv_read('out.csv')

        self.assertTrue(np.array_equal(result, control))

    def test_3_stage(self):
        from sklearn.preprocessing import Imputer

        infile_name = path_of_data('missing_vals.csv')

        p = Pipeline()

        csv_read_node = p.add(CSVRead(infile_name))
        csv_write_node = p.add(CSVWrite(self._tmp_files.get('out.csv')))
        impute_node = p.add(wrap_and_make_instance(Imputer))

        csv_read_node['output'] > impute_node['X_train']
        impute_node['X_new'] > csv_write_node['input']

        p.run()

        ctrl_imputer = Imputer()
        ctrl_X_sa = np.genfromtxt(infile_name, dtype=None, delimiter=",",
                                  names=True)
        num_type = ctrl_X_sa[0][0].dtype
        ctrl_X_nd, ctrl_X_sa_type = np_sa_to_nd(ctrl_X_sa)
        ctrl_X_new_nd = ctrl_imputer.fit_transform(ctrl_X_nd)
        control = ctrl_X_new_nd

        result = self._tmp_files.csv_read('out.csv', True)

        self.assertTrue(np.allclose(result, control))

    def test_DAG(self):
        p = Pipeline()

        s0 = OneCellLambdaStage(lambda: 'S0')
        s1 = OneCellLambdaStage(lambda: 'S1')
        s2 = OneCellLambdaStage(lambda: 'S2')
        s3 = OneCellLambdaStage(lambda x, y: '({},{})->I{}'.format(x, y, '3'))
        s4 = OneCellLambdaStage(lambda x, y: '({},{})->I{}'.format(x, y, '4'))
        s5out = StringIO()
        s6out = StringIO()
        s5 = OneCellLambdaStage(lambda x, y: '({},{})->T{}'.format(x, y, '5'),
                         fout=s5out)
        s6 = OneCellLambdaStage(lambda x: '({})->T{}'.format(x, '6'),
                         fout=s6out)
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

    def test_integrate(self):
        p_outer = Pipeline()
        p_inner = Pipeline()

        out0 = OneCellLambdaStage(lambda: 'hamster,elderberry')
        out1 = OneCellLambdaStage(lambda x: ''.join(sorted(x.replace(',', ''))) +
                           '_out1')
        sio = StringIO()
        out2 = OneCellLambdaStage(lambda x, y: '[{},{}]'.format(x, y), fout=sio)

        in0 = OneCellLambdaStage(lambda x: x.split(','), n_results=2)
        in1 = OneCellLambdaStage(lambda x: ''.join(sorted(x)) + '_in1')
        in2 = OneCellLambdaStage(lambda x: ''.join(sorted(x)) + '_in2')
        in3 = OneCellLambdaStage(lambda x, y: '({},{})'.format(x, y))

        in_nodes = [p_inner.add(s) for s in (in0, in1, in2, in3)]
        out_nodes = [p_outer.add(s) for s in (out0, out1, out2)]

        in_nodes[0]['fx0'] > in_nodes[1]['x']
        in_nodes[0]['fx1'] > in_nodes[2]['x']
        in_nodes[1]['fx'] > in_nodes[3]['x']
        in_nodes[2]['fx'] > in_nodes[3]['y']

        in_node_proxy = p_outer._Pipeline__integrate(None, p_inner, 
                                                     in_nodes[0],
                                                     in_nodes[3])

        out_nodes[0]['fx'] > in_node_proxy['x']
        out_nodes[0]['fx'] > out_nodes[1]['x']
        in_node_proxy['fx'] > out_nodes[2]['x']
        out_nodes[1]['fx'] > out_nodes[2]['y']

        p_outer.run()

        control = '[(aehmrst_in1,bdeeelrrry_in2),abdeeeehlmrrrrsty_out1]'

        self.assertEqual(sio.getvalue(), control)

    def test_syntax_iss48(self):
        # https://github.com/dssg/UPSG/issues/48
        stage_in = MockupStage((), ('output',))
        stage_trans = MockupStage(('input',), ('output',))
        stage_filter = MockupStage(('input',), ('output', 'complement'))
        stage_split_y = MockupStage(('input',), ('X', 'y'))
        stage_clf = MockupStage(('X_train', 'X_test', 'y_train'), ('y_pred', 'params'))
        stage_out = MockupStage(('result', 'params'), ())

        p_ctrl = Pipeline()
        p_ctrl_in = p_ctrl.add(stage_in, 'in')
        p_ctrl_trans = p_ctrl.add(stage_trans, 'trans')
        p_ctrl_filter = p_ctrl.add(stage_filter, 'filter')
        p_ctrl_split_y_test = p_ctrl.add(stage_split_y, 'split_y_test')
        p_ctrl_split_y_train = p_ctrl.add(stage_split_y, 'split_y_train')
        p_ctrl_clf = p_ctrl.add(stage_clf, 'clf')
        p_ctrl_out = p_ctrl.add(stage_out, 'out')

        p_ctrl_in['output'] > p_ctrl_trans['input']
        p_ctrl_trans['output'] > p_ctrl_filter['input']
        p_ctrl_filter['output'] > p_ctrl_split_y_train['input']
        p_ctrl_filter['complement'] > p_ctrl_split_y_test['input']
        p_ctrl_split_y_train['X'] > p_ctrl_clf['X_train']
        p_ctrl_split_y_train['y'] > p_ctrl_clf['y_train']
        p_ctrl_split_y_test['X'] > p_ctrl_clf['X_test']
        p_ctrl_clf['y_pred'] > p_ctrl_out['result']
        p_ctrl_clf['params'] > p_ctrl_out['params']

        p_result = Pipeline()
        p_result_in = p_result.add(stage_in, 'in')
        p_result_trans = p_result.add(stage_trans, 'trans')
        p_result_filter = p_result.add(stage_filter, 'filter')
        p_result_split_y_test = p_result.add(stage_split_y, 'split_y_test')
        p_result_split_y_train = p_result.add(stage_split_y, 'split_y_train')
        p_result_clf = p_result.add(stage_clf, 'clf')
        p_result_out = p_result.add(stage_out, 'out')

        p_result_in > p_result_trans 
        p_result_filter(p_result_trans)
        p_result_split_y_train(p_result_filter)
        p_result_split_y_test(p_result_filter['complement'])
        p_result_clf(
                X_train=p_result_split_y_train['X'], 
                y_train=p_result_split_y_train['y'],
                X_test=p_result_split_y_test['X'])
        p_result_out(p_result_clf['y_pred'], p_result_clf['params'])

        self.assertTrue(p_ctrl.is_equal_by_str(p_result))
if __name__ == '__main__':
    unittest.main()
