import numpy as np

from itertools import Counter

def value_counts(a_list):
    """counter, but pandas syntax
    """
    return Counter(a_list)

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