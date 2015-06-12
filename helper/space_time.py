from collections import namedtuple
from tuples import GPS
import numpy as np

from sklearn import cross_validation



def distance(gps_1, gps_2):
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
    r = 3956 # 6371 Radius of earth in kilometers. Use 3956 for miles
    return c * r
    
def dist_less_than(gps_1, gps_2, threshold):
    """single line description
    Parameters
    ----------
    val : float
       miles 

    Returns
    -------
    boolean 
    
    """
    return (distance(gps_1, gps_2) < threshold)
    
def time_less_then(time_1, time_2, val):
    #check both in datetime64
    return (time_2-time_1 < val) and  (time_2 - time_1 >= 0 ) 

def select_by_dist_from(data, target, threshold, lat_col_name, lng_col_name):
    ret = []
    for idx, x in enumerate(data):        
        if dist_less_than(target, GPS(x[lng_col_name], x[lat_col_name]),threshold):
            ret.append(idx)
    return ret
    
def select_by_time_from(data, target, threshold, time_col_name):
    ret = []
    for idx, x in enumerate(data):
        if time_less_then(target, x[time_col_name]):
            ret.apped(idx)
    return ret

#args = [('dist', ldoughnut, (10, 20), GPS(0, 0)),
#        ('color', lcolor, ('red', 'blue'), None)]

#def ldoughnet(cols, run_args, fixed_args):
#   origin = fixed_args
#   inner_radius = run_args
#   return np.and(dist_less(col[0], inner_radius + 1, origin) np.not(dist_less(col[1], inner_radius, origin)))



def select_one(lambdas, col_names, data)
   to_select = vector_of_trues(data.rows)
   for lambda, col_name in zip(lambdas, col_names):
       to_select = np.and(to_select, lambda(data[col_name])
   return to_select

def select_all(args, data):
   changing_args = [arg[2] for arg in args]
   runs = product(*changing_args)
   for r in runs:
      fixed_lambdas = [lambda col: lam(col, r[idx], args[idx][3]) for idx, lam in enumerate(lambdas)]
      yield select_one(fixed_lambdas, args[idx][1], data)
      



data = np.array([ ('home', 40.761036, -73.977374),
                  ('work', 45.5660930, -73.92599),
                  ('fun', 40.702646, -74.013799)],
                  dtype = [('name', 'S4'), ('lng', float), ('lat', float)]
                )    
                
target = GPS(40.748784, -73.985429)
threshold = .001

lat_col_name = 'lat'
lng_col_name = 'lng'

data_id =  select_by_dist_from(data, target, threshold, lat_col_name, lng_col_name)


import pdb; pdb.set_trace()

       