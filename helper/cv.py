
def accuracy_score_wrapper(clf, X_test, y_test, **kwargs):
    return accuracy_score(y_test, clf.predict(X_test), **kwargs)

def log_loss_wrapper(clf, X_test, y_test, **kwargs):
    return log_loss(y_test, clf.predict(X_test), **kwargs)

def precision_recall_curve_wrapper(clf, X_test, y_test, **kwargs):
    return precision_recall_curve(y_test, clf.predict_proba(X_test), **kwargs)

def recall_score_wrapper(clf, X_test, y_test, **kwargs):
    return recall_score(y_test, clf.predict(X_test), **kwargs)

def precision_score_wrapper(clf, X_test, y_test, **kwargs):
    return precision_score(y_test, clf.predict(X_test), **kwargs)

def hinge_loss_wrapper(clf, X_test, y_test, **kwargs):
    return hinge_loss(y_test, clf.decision_function(X_test), **kwargs)


def cv_by_time_range_widening(data, start_time, first_len, increment_by, repetitions, time_col_name):
    cur_len = first_len
    for _ in range(repetitions):
        yield (select_by_time_from(data, start_time, cur_len, time_col_name),
               select_by_time_from(data, start_time + cur_len, increment_by, time_col_names)
        cur_len += increment_by
       
def cv_by_time_range_fixed(data, start_time, first_len, increment_by, repetitions, time_col_name):
    cur_len = first_len
    for _ in range(repetitions):
        yield (select_by_time_from(data, start_time, cur_len, time_col_name),
            select_by_time_from(data, start_time + cur_len, increment_by, time_col_names)
        start_time += increment_by     

def get_n_top_features(clf, n_top):
    if hasattr(clf, 'feature_importances_'): 
        return ( [ clf.feature_importances_[x] for
                x in clf.feature_importances_.argsort()[-n_top:][::-1]])

         

def get_n_top_features(clf, n_top):
    if hasattr(clf, 'feature_importances_'): 
        return ( [ clf.feature_importances_[x] for
                x in clf.feature_importances_.argsort()[-n_top:][::-1]])


def run_cv(clf, X, y, cv_function, cv_inputs, analysis_funcs, analysis_kwargs):
    """
    
    Parameters:
    -----------
    cv_function : {cv_by_time_range_widening, cv_by_time_range_fixed}
    analysis_funcs : list of ((sklearn.BaseEstimator, numpy.ndarray, numpy.ndarray, dict) -> numpy.ndarray)
        A list of functions that process:
            1. A clf fitted with the appropriate training and testing data for this round of cv
            2. The X test data
            3. The y test data
            4. Some keyword arguments
        And returns a numpy array
    analysis_kwargs : list of dict
        kwargs to pass to analysis funcs.
    """
    
    ret = []
    for train, test in cv_function(data, **cv_inputs):
        
        clf.fit(X[train],y[train])
        
        sub_list = []
        
        for idt, t in enumerate(analysis_funcs):
            sub_list.append( t(clf, X[test], y[test], **analysis_kwargs[idt] ) ) 
    
        ret.append(r)
    
    return ret
    