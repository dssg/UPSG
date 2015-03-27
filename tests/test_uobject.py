import numpy as np
from os import system
import unittest

from upsg.uobject import *
from upsg.utils import np_type
from utils import path_of_data, UPSGTestCase


class TestUObject(UPSGTestCase):
    test_dict = {'k1': 'A String that is fairly long', 'k2': 42.1, 'k3': 7}
    test_dict_keys = test_dict.keys()
    test_array_dtype = np.dtype({'names': test_dict_keys, 'formats': [
                                np_type(test_dict[key]) for key in
                                test_dict_keys]})
    test_array_vals = [tuple([test_dict[key] for key in test_dict_keys])]
    test_array = np.array(test_array_vals, dtype=test_array_dtype)

    def test_csv_load_store(self):
        filename = path_of_data('mixed_csv.csv')
        uo = UObject(UObjectPhase.Write)
        uo.from_csv(filename)
        uo.write_to_read_phase()
        result = uo.to_np()
        control = np.genfromtxt(
            filename,
            dtype=None,
            delimiter=",",
            names=True)
        self.assertTrue(np.array_equal(result, control))

    def test_dict_load_store(self):
        d = self.test_dict
        uo = UObject(UObjectPhase.Write)
        uo.from_dict(d)
        uo.write_to_read_phase()
        result = uo.to_dict()
        self.assertEqual(d, result)

    def test_dict_to_np(self):
        uo = UObject(UObjectPhase.Write)
        uo.from_dict(self.test_dict)
        uo.write_to_read_phase()
        A = uo.to_np()
        self.assertTrue(np.array_equal(self.test_array, A))
        self.assertEqual(self.test_array.dtype, A.dtype)

    def test_np_to_dict(self):
        uo = UObject(UObjectPhase.Write)
        uo.from_np(self.test_array)
        uo.write_to_read_phase()
        d = uo.to_dict()
        self.assertEqual(d, self.test_dict)

    def test_sql(self):
        db_url = 'sqlite:///{}'.format(path_of_data('small.db'))
        for tbl_name in ('employees', 'hours'):
            uo_in = UObject(UObjectPhase.Write)
            uo_in.from_sql(db_url, {}, tbl_name, False)
            uo_in.write_to_read_phase()
            sa = uo_in.to_np()
            uo_out = UObject(UObjectPhase.Write)
            uo_out.from_np(sa)
            uo_out.write_to_read_phase()
            tbl_result, conn_result, db_url, conn_params = uo_out.to_sql(
                db_url, {}, '{}_out'.format(tbl_name))
            result = conn_result.execute(
                sqlalchemy.sql.select([tbl_result])).fetchall()
            tbl_result.drop(conn_result)

            ctrl_engine = sqlalchemy.create_engine(db_url)
            md = sqlalchemy.MetaData()
            md.reflect(ctrl_engine)
            tbl_ctrl = md.tables[tbl_name]
            ctrl = ctrl_engine.execute(
                sqlalchemy.sql.select([tbl_ctrl])).fetchall()

            self.assertEqual(result, ctrl)

if __name__ == '__main__':
    unittest.main()
