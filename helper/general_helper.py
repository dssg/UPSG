import matplotlib.pyplot as plt

import numpy as np
from collections import Counter
from collections import namedtuple

from itertools import permutations
from itertools import product

import struct 
import re

import datetime
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import random
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score

GPS = namedtuple('GPS',['lng', 'lat'])

class UtilityError(Exception):
    """docstring for UtilityError"""
    pass


def spaces_to_snake(a_string):
    """converts a string that has spaces into snake_case
    Example:
        print camel_to_snake("KENNY BROUGHT HIS WIFE")
        > KENNY_BROUGHT_HIS_WIFE
    """
    s = re.sub(r"\s+", '_', a_string)
    return s.lower()

def camel_to_snake(a_string):
    """convert Camel Case to snake_case 
     http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case  
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def random_timedelta(low, high, size):
    """returns list of random time deltas
    ERIC's Function
    Parameters
    ----------
    low : int
       Description 
    high : int
       Description 
   size : int
      Description
       
    Returns
    -------
    List
       List of timedelta objects
        
    """
    random_ints = np.random.randint(low, high, size=size)
    return [datetime.timedelta(random_ints[i]) for i in range(size)]

def random_dates(start, end, size):
    """single line description
    ERIC's Function
    
    Parameters
    ----------
    start : datetime
       Description 
    end : datetime
       Description
    size : int
       Description????????????
      
    Returns
    -------
    List
       List of random dates
       
    """
    random_time = random_timedelta(0, (end-start).days, size)
    return [size+random_time[i] for i in range(size)]
    



def plot_marginal_precision_curve(labels, probas_pred, **kwargs):
    """Returns marginals (% true / total students with that prob)
    CARL's Function
    
    Parameters
    ----------
    temp : type
       Description 
    
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    Returns marginals (% true / total students with that prob) and plots it
    against predicted probabilities. A straight line is good.
           
    """
    #check if nparray
    if not isinstance(labels, np.ndarray):
        if isinstance(labels, list):
            labels = np.array(labels)
        else:
            raise UtilityError("labels not list or ndarray")
    if not isinstance(probas_pred, np.ndarray):
        if isinstance(probas_pred, list):
            probas_pred = np.array(probas_pred)
        else:
            raise UtilityError("probas_pred not list or ndarray")        
            

    
    totals = [probas_pred[np.where(probas_pred==p)].sum() for p in probas_pred]
    trues = [lables[np.where(probas_pred==p)].sum() for p in probas_pred]
    
    marginals = [float(true)/total for true, total in zip(trues, totals) if total != 0]
    plt.figure()
    plt.plot(predicted_probs, marginals, **kwargs)
    plt.title('Marginal Precision Curve')
    plt.xlabel('Predicted Probs')
    plt.ylabel('Actual Proportions')
    plt.show()
    return totals, trues, marginals        
    
def plot_curve(curve_type, y_test, probas_pred, clf_name='', plot_curve=True, **kwargs):
    """Plot either a ROC curve or Precision and Recall
    CARL's Function
    
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
    auc, x, y, thresholds
    
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
    """returns top features and values from sklearn clf object
    
    Parameters
    ----------
    Clf : SK-learn model
       trained model 

    n_top : int
       specify the n_top values
    
    Returns
    -------
    List 
       [(col_num, val), ... ]
   
    """
    if hasattr(clf, 'feature_importances_'): 
        return ( [ (x,clf.feature_importances_[x]) for
                x in clf.feature_importances_.argsort()[-n_top:][::-1]])
    raise UtilityError("CLF either not fit, or model does not have feature imporances")
    
def identicle_value_columns(M, value, orientation):
    """Checks for identical values in row or column in a numpy object
    Parameters
    ----------
    M : numpy.array
       The matrix to check 
    
    orientation : str or int
    
    value : int
        the value to match against
        
    Returns
    -------
    List
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
        
def value_counts_top(L, n): 
    """count elements in list 
    Parameters
    ----------
    L : list
       A list of elements to be counted 
    n : int
       Number of top values   
       
    Returns
    -------
    List 
        A list of tuples (value, count)
       
    """
    return Counter(L).most_common(n)
    
def value_counts(a_list):
    """counter, but pandas syntax
    """
    return Counter(a_list)

def describe(a_list):
    """gives count, mean, std, min, and max from list
    """
    cnt = len(a_list)
    mean = np.mean(np.array(a_list))
    std = np.std(np.array(a_list))
    mi = min(a_list)
    mx = max(a_list)    
    return cnt, mean, std, mi, mx
    
    
def crosstab(list_1, list_2):
    """Contingancy table: compute a frequency table from two lists
    Parameters
    ----------
    temp : type
       Description 
    
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    list_1 = np.array(list_1)
    list_2 = np.array(list_2)
    
    key_1 = np.unique(list_1)
    key_2 = np.unique(list_2)
    
    a_dict = {}
    
    for aKey in key_1:
        loc_aKey = np.where(np.array(list_1)==aKey)[0]
        tmp_list = list_2[loc_aKey]
        cnt = Counter(tmp_list)
        a_dict[aKey] = cnt
    
    M = np.zeros(shape=(len(key_1),len(key_2)))
    for idx, x in enumerate(key_1):
        for idy, y in enumerate(key_2):
            M[idx,idy] = a_dict[x][y]
    print_Matrix_row_col(M, key_1, key_2)
    return a_dict
    
def print_Matrix_row_col( M, L, L_2,):
    """single line description
    Parameters
    ----------
    L : List 
       Description 
    M : List 
       Description
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    row_format ="{:>15}" * (len(L_2) + 1)
    print row_format.format("", *L_2)
    for team, row in zip(L, M):
        print row_format.format(team, *row)
    return None

def open_csv(loc):
    """single line description
    Parameters
    ----------
    temp : type
       Description 
    
    Attributes
    ----------
    temp : type
       Description 
       
    Returns
    -------
    temp : type
       Description
       
    """
    with open(loc, 'rb') as f:
        reader = csv.reader(f)
        data= list(reader)
    return data
    
    
def save_file(data, destination):
    with open(destination, 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(data)
    return

def convert_fixed_width_list_to_CSV_list(data, list_of_widths):
    s = "s".join([str(s) for s in list_of_widths])
    s= s + 's'
    out = []
    for x in data:
        out.append(struct.unpack(s, x[0]))
    return out


def run_strat_cross(clf, M, truth, n_folds, type_of):
    skf = cross_validation.StratifiedKFold(truth, n_folds=n_folds)
    acc = []
    for train_index, test_index in skf:  
        X_train, X_test = M[train_index], M[test_index]
        y_train, y_test = truth[train_index], truth[test_index]
        clf.fit(M[train_index], truth[train_index])
        
        if type_of == 'acc':
            acc.append(accuracy_score(truth[test_index], clf.predict(M[test_index])))
        elif type_of == 'feat':
             acc.append(clf.feature_importances_) 
                     
    return acc


def random_strat_cv(clf, M, truth, subsample_size, n_folds):
    """single line description
    Parameters
    ----------
    clf : type
       the Random Forest 
    M : type
       the Random Forest 
    r : type
       the Random Forest
    clf : type
       the Random Forest      
    Returns
    -------
    temp : type
       Description
       
    """
    acc = []
    
    subset = random_indicies(len(truth), subsample_size)
    
    truth_t = np.concatenate((np.zeros(subsample_size),np.ones(subsample_size)), axis=1)     
    
    skf = cross_validation.StratifiedKFold(truth_t, n_folds=n_folds)

    for train_index, test_index in skf:  
        X_train, X_test = M[train_index], M[test_index]
        y_train, y_test = truth[train_index], truth[test_index]
        print("TRAIN:", train_index, "TEST:", test_index)
        
        #train = [subset[ind] for ind in train_ind]
        #test = [subseload_digitst[ind] for ind in test_ind]
        #print train, test
        clf.fit(M[train_index], truth[train_index])
        
        print clf.feature_importances_
        
        import pdb; pdb.set_trace()
        #acc.append(accuracy_score(truth[test], clf.predict(M[test])))
    return 

def random_from_list(a_list, cnt):
    """single line description
    Parameters
    ----------
    a_list : list
      the length of the different types.  
    cnt : int
      the size of the subsample 
   
    Returns
    -------
    result : list
       Location of indices #TODO
       
    """
    index = 0
    result = []
    for x in a_list:
        result.append(random.sample(range(index, index + x), cnt))
        index += x
    return [x for b in result for x in b]


def repeat_random_from_list(a_list, cnt, length):
    'makes a list of random ranges'
    result = []
    for x in range(length):
        result.append(random_from_list(a_list, cnt))
        
    return result

def choose_n_from_categories(target, n):
    """single line description
    Parameters
    ----------
    temp : type
       Description
       
    Returns
    -------
    temp : type
       Description
       
    """
    #check if target is np.array
    target = np.array(target)
    catagories = np.unique(target)
    result =[]
    for x in catagories:
        result.append(random.sample( np.where(target==x)[0], n)  )
    return results

def haversine(gps_1, gps_2):
    from math import radians, cos, sin, asin, sqrt
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    from:
    http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # convert decimal degrees to radians 
    lon1, lat1 = gps_1.lng, gps_1.lat
    lon2, lat2 = gps_2.lng, gps_2.lat
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def dist_less_than(gps_1, gps_2, val):
    return (haversine(gps_1, gps_2)<val)

def time_less_then(time_1, time_2, val):
    #check both in datetime64
    return (time_2-time_1 < val)
    
def select_by_function(origin, data, a_func, threshold):
    a_list= []
    for idx,x in data:
        if a_func(origin, x)==True:
            a_list.append(idx)
    return a_list



data = np.array([ ('home', 40.761036, -73.977374),
                  ('work', 40.760930, -73.992599),
                  ('fun', 40.702646, -74.013799)],
                  dtype = [('name', 'S4'), ('lng', float), ('lat', float)]
                )    
target = GPS._make((40.748784, -73.985429))
threshold = 1

lat = 'lat'
lng = 'lng'


 
import pdb; pdb.set_trace()


test_gps = [(40.761036, -73.977374),
            (40.760930, -73.992599),
            (40.702646, -74.013799),
            (40.674026, -73.944533),
            (40.673098, -73.943997),
            (40.674872, -73.943418),
            (40.674009, -73.945027)
            ]


test_gps_list = [GPS._make(x) for x in test_gps]
target = GPS._make((40.748784, -73.985429))


print [dist_less_than(x,target,1) for x in test_gps_list]
    
import pdb; pdb.set_trace()

    

#choose_n_from_categories([1,1,1,1,3,3,3,4,4,4], 2)




def test_rf():
    from sklearn import datasets
    digit = datasets.load_digits()
    #What are the most discripimitive feautres between 1 and 2
    zeros_data = []
    zeros_target = []
    
    for idx, x in enumerate(digit.target):
        if x == 0:
            zeros_target.append(x)
            zeros_data.append(digit.data[idx])
    one_data = []
    one_target = []
    for idx, x in enumerate(digit.target):
        if x == 1:
            one_target.append(x)
            one_data.append(digit.data[idx])
    one_data = one_data[:-4]
    one_target = one_target[:-4]
    # thiss is 
    for x in one_data:
        zeros_data.append(x)
    for x in one_target:
        zeros_target.append(x)
    
    #now we want to know how many we need to train on to get a good fit.
    clf = RandomForestClassifier()
    clf.fit(zeros_data,zeros_target)
    clf.feature_importances_
    #How accuarte are we?
    for x in repeat_random_from_list([178,178], 100, 10):
        result = run_strat_cross(clf, np.array(zeros_data)[x], np.array(zeros_target)[x], 10, 'acc' )
        print sum(result)
    
    import pdb; pdb.set_trace()


    

test_rf()
import pdb; pdb.set_trace()

ran = random_indicies(1797, 100)


result = run_strat_cross(clf, digit.data, digit.target,  3, 'acc')

import pdb; pdb.set_trace()
#random_strat_cv(clf, np.array([[1,2,3],[1,2,4],[1,2,3],[3,4,5]]), [0,0,1,1], 2, 2)
#clf.feature_importances_

















    
import pdb; pdb.set_trace()