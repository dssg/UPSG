import matplotlib.pyplot as plt

import numpy as np
from collections import Counter
from collections import namedtuple

from itertools import permutations
from itertools import product

import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class UtilityError(Exception):
    """docstring for UtilityError"""
    pass
        
    
def plot_curve(curve_type, y_test, probas_pred, clf_name='', plot_curve=True, **kwargs):
    """Plot either a ROC curve or Precision and Recall
    Parameters
    ----------
    curve_type : str
       either 'ROC' or 'PR' 
    
    y_test : list or array
        Table of correct Labels
        
    probas_pred : list or array
        Predicted probabilities for array of values
        
    clf_name : str
        Name of example
    
    plot_curve : Boolean 
        Either plot or not the curve
        
    **kwargs : 
        To pass to the plt.plot()
        
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    if curve_type == 'ROC':
        x, y, thresholds = roc_curve(y_test, probas_pred, pos_label=1)
        xlabel = 'False Positive Rate or (1 - Specificity)'
        ylabel = 'True Positive Rate'
        title = 'ROC Curve - {}'.format(clf_name)
    if curve_type == 'PR':
        x, y, thresholds = precision_recall_curve(y_test, probas_pred)
        xlabel = 'Recall'
        ylabel = 'Precision'
        title = 'Precision-Recall Curve - {}'.format(clf_name)
        
    auc_temp = auc(x, y)
    if plot_curve:
        plt.figure()
        plt.plot(x, y, label='Curve area = {:.2f}'.format(auc_temp), **kwargs)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='best')
        
    return auc, x, y, thresholds
    
def top_features_and_values(clf, n_top):
    """returns top features and values
    
    Parameters
    ----------
    Clf : SK-learn model
       trained model 

    n_top : int
       specify the n_top values
    
    Returns
    -------
    results : list of tuples 
       [(col_num, val), ... ]
   
    """
    if hasattr(clf, 'feature_importances_'): 
        return ( [ (x,clf.feature_importances_[x]) for
                x in clf.feature_importances_.argsort()[-n_top:][::-1]])
    raise UtilityError("CLF either not fit, or model does not have feature imporances")
    
def identicle_value_columns(M, value, orientation):
    """Checks for identical values in row or column.
    Parameters
    ----------
    M : numpy.array
       The matrix to check 
    
    orientation : str or int
    
    value : int
        the value to match against
        
    Returns
    -------
    result : list
       row or column index of identical values
       
    """
    if type(orientation) is str:
        if orientation == 'col':
            return np.where(np.all(M==value, axis=0))
        elif orientation == 'row':
            return np.where(np.all(M==value, axis=1))
    elif type(orientation) is int:
        return np.where(np.all(M==value, axis=orientation))
    else:
        raise UtilityError("Orientation is improperly formatted.")    