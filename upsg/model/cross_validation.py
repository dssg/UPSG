
import itertools as it
import numpy as np

from ..stage import RunnableStage, MetaStage
from ..uobject import UObject, UObjectPhase
from ..pipeline import Pipeline
from ..transform.split import KFold
from ..transform.identity import Identity
from ..wrap.wrap_sklearn import wrap


class CrossValidationScore(MetaStage):

    """Performs KFold cross-validation on a given classifier Stage and 
    outputs the average score

    **Input Keys**

    X_train

    y_train

    **Output Keys**

    score

    Parameters
    ----------
    clf_stage : Stage class
        class of a Stage to cross-validate
    clf_stage_kwargs : dict of str: ? (default {})
        The parameters pass to the classifier
    score_key : str
        key output from clf_stage that should be used for scoring. 
            The table that the key stores should be of size 1x1
    cv_stage : Stage class or None
        class of stage to provide cross-validate partitioning.
        For example, 
        upsg.wrap.wrap_sklearn.wrap('sklearn.cross_validation.KFold')
        If None, then
        upsg.wrap.wrap_sklearn.wrap('sklearn.cross_validation.KFold')
        is used.
    cv_stage_kwargs: dict of str: ?
        Arguments corresponding to the keyword arguments of
        sklearn.cross_validation.KFold including n_folds


    """

    class __ReduceStage(RunnableStage):

        def __init__(self, n_parents):
            self.__n_parents = n_parents
            self.__score_keys = map('score_in{}'.format, range(n_parents))
            self.__input_keys = self.__score_keys
            self.__output_keys = ['score']

        @property
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return self.__output_keys

        def run(self, outputs_requested, **kwargs):
            # TODO return data in a format that tells you what the params were
            scores_array = np.array(
                [kwargs[key].to_np()[0][0] for key in self.__score_keys])
            score = UObject(UObjectPhase.Write)
            score.from_np(np.mean(scores_array))
            return {'score': score}

    def __init__(self, clf_stage, clf_stage_kwargs={}, score_key='score', 
                 cv_stage=None, cv_stage_kwargs={}):

        if cv_stage is None:
            cv_stage = wrap('sklearn.cross_validation.KFold')
        p = Pipeline()
        self.__pipeline = p

        node_map = p.add(Identity(('X_train', 'y_train')))
        cv_stage_kwargs['n_arrays'] = 2
        try:
            n_folds = cv_stage_kwargs['n_folds']
        except KeyError:
            n_folds = 2
            cv_stage_kwargs['n_folds'] = 2
        node_cv = p.add(cv_stage(**cv_stage_kwargs))
        node_reduce = p.add(self.__ReduceStage(n_folds))

        node_map['X_train_out'] > node_cv['input0']
        node_map['y_train_out'] > node_cv['input1']
        if 'y' in node_cv.input_keys:
            node_map['y_train_out'] > node_cv['y']

        for fold_i in xrange(n_folds):
            node_clf = p.add(clf_stage(**clf_stage_kwargs))
            node_cv['train0_{}'.format(fold_i)] > node_clf['X_train']
            node_cv['train1_{}'.format(fold_i)] > node_clf['y_train']
            node_cv['test0_{}'.format(fold_i)] > node_clf['X_test']
            node_cv['test1_{}'.format(fold_i)] > node_clf['y_test']
            node_clf['score'] > node_reduce['score_in{}'.format(fold_i)]

        self.__in_node = node_map
        self.__out_node = node_reduce

    @property
    def input_keys(self):
        return ['X_train', 'y_train']

    @property
    def output_keys(self):
        return ['score']

    @property
    def pipeline(self):
        return (self.__pipeline, self.__in_node, self.__out_node)

