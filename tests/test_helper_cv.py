import unittest
from helper.cv import distance

from helper.tuples import GPS

class TestHelperCV(unittest.TestCase):
    def test_distance(self):
        gps_a = GPS(40.761036, -73.977374)
        gps_b = GPS(40.861036, -73.877374)
        
        self.assertTrue(distance(gps_a, gps_b)>7)
        
        
    #def test_cv(self):
    #    test_gps = [(40.761036, -73.977374),
    #                (40.760930, -73.992599),
    #                (40.702646, -74.013799),
    #                (40.674026, -73.944533),
    #                (40.673098, -73.943997),
    #                (40.674872, -73.943418),
    #                (40.674009, -73.945027)
    #                ]
    #    test_gps_list = [GPS._make(x) for x in test_gps]
    #    
    #  
    #    
    #    
    #    print [dist_less_than(x,target,1) for x in test_gps_list]
    #    
    #    import pdb; pdb.set_trace()
        

if __name__ == '__main__':
    unittest.main()