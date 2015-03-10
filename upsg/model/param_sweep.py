import itertools as it
import numpy as np

from ..stage import RunnableStage, MetaStage
from ..uobject import UObject, UObjectPhase


class ParamSweep(MetaStage):
    # TODO dynamically generate input and output keys according to the clf

    class __MapStage(RunnableStage):
    """Translates metastage input keys to input stage required by the stage"""
    # Just passes the values on for now. It might need to modify them later
        def __init__(self, n_children):
            self.__n_children = n_children
            self.__input_keys = ['X_train', 'y_train', 'X_test', 'y_test']
#            self.__output_keys_hier = {in_key : map(
#                lambda child: '{}{}'.format(in_key, child), xrange(n_children))
#                for in_key in self.__input_keys}
#            self.__output_keys = list(it.chain.from_iterable(
#                self.__output_keys_hier.values()))
            self.__output_keys = self.__input_keys

        @property
        def input_keys(self):
            return self.__input_keys

        @property 
        def output_keys(self):
            return self.__output_keys

        def run(self, outputs_requested, **kwargs):
#            ret_hier = ({out_key : kwargs[in_key] for out_key 
#                in self.__output_keys_hier[in_key]} for in_key
#                in self.__input_keys)
#            ret = {}
#            map(ret.update, ret_hier)
#            return ret    
            return kwargs
    
    class __ReduceStage(RunnableStage):
        def __init__(self, n_parents):
            self.__n_parents = n_parents
            self.__input_keys = map('score{}'.format, range(n_parents))
            self.__output_keys = ['scores']

        @property
        def input_keys(self):
            return self.__input_keys

        @property 
        def output_keys(self):
            return self.__output_keys

        def run(self, outputs_requested, **kwargs):
            #TODO return data in a format that tells you what the params were
            array = np.array(
                [kwargs[key].to_np()[0,:] for key in self.__input_keys])
            scores = UObject(UObjectPhase.Write)
            scores.from_np(array)
            return {'scores' : scores}

    def __init__(self, clf_stage, score_key, param_dict):
        #TODO respect score_key
        self.__clf_stage = clf_stage
        # produces dictionaries of the cartesian product of our parameters.
        # from http://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
        self.__params_prod = [dict(it.izip(params_dict, x)) 
            for x in it.product(*params_dict.itervalues())]
        width = len(self.__params_prod)
        p = Pipeline()
        self.__pipeline = p
        node_map = p.add(__MapStage(width))
        node_reduce = p.add(__ReduceStage(width))
        for i, params in enumerate(self.__params_prod):
            node = p.add(clf_stage(**params))
            [uid_map[key] > node[key]) for key in 
                ['X_train', 'X_test', 'y_train', 'y_test']]
            node['score'] > node_reduce['score{}'.format(i)]
        self.__in_node = node_map
        self.__out_node = node_reduce
    
    @property
    def input_keys(self):
        return ['X_train', 'y_train', 'X_test', 'y_test']

    @property
    def output_keys(self):
        return ['y_pred', 'params']

    @property
    def pipeline(self):
        return (self.__pipeline, self__in_node, self.__out_node)

    def run(self, outputs_requested, **kwargs):
        pass
        #TODO run doesn't really make sense here. Maybe we should restructure
