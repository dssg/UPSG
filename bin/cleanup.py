#!/usr/bin/env python

import sys
import os
import glob
import tables
import sqlalchemy


def run(path='.'):
    """removes .upsg files and temporary sql tables from
    the given path"""

    for file in glob.iglob(os.path.join(path, '*.upsg')):
        hfile = tables.open_file(file, mode='r')
        storage_method = hfile.get_node_attr('/upsg_inf', 'storage_method')
        if storage_method == 'sql':
            sql_group = hfile.root.sql
            pipeline_generated = hfile.get_node_attr(sql_group,
                                                     'pipeline_generated')
            if pipeline_generated:
                db_url = hfile.get_node_attr(sql_group, 'db_url')
                tbl_name = hfile.get_node_attr(sql_group, 'tbl_name')
                conn_params = np_sa_to_dict(hfile.root.sql.conn_params.read())
                engine = sqlalchemy.create_engine(db_url)
                conn = engine.connect(**conn_params)
                md = sqlalchemy.MetaData()
                md.reflect(conn)
                tbl = md.tables[tbl_name]
                tbl.drop(conn)
        hfile.close()
        os.remove(file)


def usage():
    print """cleanup.py - removes .upsg files and temporary sql tables from
    a given directory.

    usage: cleanup.py [dir]
       or: cleanup.py --help displays this message
       or: cleanup.py -h     displays this message

    Arguments:
       dir: The directory to clean. Is the current workind directory
            by default.
    """
    exit(0)

if __name__ == '__main__':
    path = '.'
    if len(sys.argv) > 1:
        if sys.argv[1] in ('-h', '--help'):
            usage()
        path = sys.argv[1]
    if not os.path.isdir(path):
        usage()
    run(path)
