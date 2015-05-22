from collections import namedtuple
import uuid

from ..stage import RunnableStage, MetaStage
from ..uobject import UObject, UObjectPhase
from ..pipeline import Pipeline
from ..utils import dict_to_np_sa
from ..wrap.wrap_sklearn import wrap, wrap_and_make_instance
from ..export.plot import Plot
from ..transform.identity import Identity

VisualMetricSpec_ = namedtuple('VisualMetricSpec', ['metric', 
                                                    'output_key_x',
                                                    'output_key_y',
                                                    'graph_title',
                                                    'graph_x_label',
                                                    'graph_y_label'])
class VisualMetricSpec(VisualMetricSpec):
    """Specification for a metric to be used with MultiMetric_. 
    
    In contrast with NumericMetricSpec_, these metrics will be reported as
    a plot rather than a number or a table.

    Attributes
    ----------
    metric : str
        The fully qualified package name of the sklearn metric: e.g.:
        'sklearn.metrics.precision_recall_curve'
    output_key_x : str
        The output key of upsg.wrap.wrap_sklearn.wrap(metric) corresponding
        the the x-axis on the graph. e.g.:
        'recall'
    output_key_y : str
        The output key of upsg.wrap.wrap_sklearn.wrap(metric) corresponding
        the the y-axis on the graph. e.g.:
        'precision'
    graph_title : str
        The title of the graph. e.g. : 'Precision/Recall Curve'
    graph_x_label : str
        The label for the graph's x-axis. e.g.: 'Recall'
    graph_y_label : str
        The label of the graph's y-axis. e.g.: "Precision"
    """
    pass


NumericMetricSpec_ = namedtuple('NumericMetricSpec', ['metric',
                                                      'output_key',
                                                      'title'])


class NumericMetricSpec(NumericMetricSpec_):
    """Specification for a metric to be used with MultiMetric_

    In contrast with VisualMetricSpec_, these metrics will be reported as a 
    number or a table rather than a plot

    Attributes
    ----------
    metric : str
        The fully qualified package name of the sklearn metric: e.g.:
        'sklearn.metrics.roc_curve'
    output_key : str
        The output key of upsg.wrap.wrap_sklearn.wrap(metric) that will be
        reported. e.g.: 'auc'
    title : str
        The title to associate with the score. e.g.: 'ROC AUC Score'

    """
    pass

class Multimetric(MetaStage):
    """
    
    A stage that automatically runs a number of metrics, makes plots, and 
    compiles them into a report. The output of a wrapped estimator
    (as with upsg.wrap.wrap_sklearn.wrap) can be fed to the input of a 
    MultiMetric, and then MultiMetric will compile a report with a number
    of metrics of that estimator's performance.

    **Input Keys**

    params
        The parameters associated with the estimator, likely available in
        the estimators "params_out" key. This will be added to the report.
    pred_proba
        Corresponds to the estimator's "pred_proba" output key
    y_true
        The true y that the estimator was attempting to predict.

    **Output Keys**
    report_file
        The file where the report has been generated

    Parameters
    ----------
    metrics : list of (VisualMetricSpec or NumericMetricSpec)
        The metrics to run. Each entry of the list corresponds to one metric
    title : str
        The title of the report
    file_name : str
        The location in which to write the report. If not provided, a random
        name will be chosen

    """

    class __ReduceStage(RunnableStage):

        def __init__(self, metrics, title, file_name):
            self.__file_name = file_name
            self.__title = title
            self.__n_metrics = len(metrics)
            self.__metrics = metrics
            self.__input_keys = (['params'] + 
                                 ['metric{}_in'.format(i) for 
                                  i in xrange(self.__n_metrics)])   

            self.__output_keys = ['report_file']

        @property
        def input_keys(self):
            return self.__input_keys

        @property
        def output_keys(self):
            return self.__output_keys


        def run(self, outputs_requested, **kwargs):
            # TODO sanitize html
            # TODO use dbg printer's table printing
            with open(self.__file_name, 'w') as fout:
                fout.write(
                        '<h3>{}</h3><h4>Best params</h4>\n<p>{}</p>\n'.format(
                            self.__title, 
                            kwargs['params'].to_dict()))
                for i, metric in enumerate(self.__metrics):
                    uo = kwargs['metric{}_in'.format(i)]
                    if isinstance(metric, VisualMetricSpec):
                        fout.write(
                            '<h4>{}</h4><p><img src="{}"/></p>\n'.format(
                                    metric.graph_title,
                                    uo.to_external_file()))
                    else:
                        fout.write(
                            '<h4>{}</h4>\n<p>{}</p>\n'.format(
                                    metric.title,
                                    uo.to_np()))

            uo_report_file = UObject(UObjectPhase.Write)
            uo_report_file.from_external_file(self.__file_name)
            return {'report_file': uo_report_file}

    def __init__(self, metrics, title, file_name=None):

        p = Pipeline()
        self.__pipeline = p
        if file_name is None:
            file_name = str(uuid.uuid4()) + '.html'

        self.__file_name = file_name

        node_map = p.add(Identity(('params', 'pred_proba', 'y_true')))
        node_reduce = p.add(self.__ReduceStage(metrics, title, file_name))

        node_map['params_out'] > node_reduce['params']

        for i, metric in enumerate(metrics):
            stage_metric = wrap_and_make_instance(metric.metric)
            in_keys = stage_metric.input_keys
            # TODO develop this to support more metrics. Maybe share code w/
            # test_wrap
            node_metric = p.add(stage_metric)
            if 'y_true' in in_keys:
                node_map['y_true_out'] > node_metric['y_true']
            if 'y_score' in in_keys:
                node_map['pred_proba_out'] > node_metric['y_score']
            if 'probas_pred' in in_keys:
                node_map['pred_proba_out'] > node_metric['probas_pred']

            metric_in_key = 'metric{}_in'.format(i)
            if isinstance(metric, VisualMetricSpec):
                out_file = '{}.png'.format(uuid.uuid4())
                node_plot = p.add(Plot(
                    out_file,
                    xlabel = metric.graph_x_label,
                    ylabel = metric.graph_y_label))
                node_metric[metric.output_key_x] > node_plot['x']
                node_metric[metric.output_key_y] > node_plot['y']
                node_plot['plot_file'] > node_reduce[metric_in_key]
            else:
                node_metric[metric.output_key] > node_reduce[metric_in_key]

        self.__in_node = node_map
        self.__out_node = node_reduce

    @property
    def input_keys(self):
        return self.__in_node.input_keys

    @property
    def output_keys(self):
        return self.__out_node.output_keys

    @property
    def pipeline(self):
        return (self.__pipeline, self.__in_node, self.__out_node)
