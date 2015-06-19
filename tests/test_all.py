#!/usr/bin/env python

from __future__ import print_function

import unittest
import os
import sys

import utils

def usage():
    print("""
usage: 
    python test_all --help | -h
    python test_all [-t]
    
    -t runs test suite in thorough mode--tests multiple pipeline run methods
       and debug output methods. When the -t is omitted. Only uses default run
       mode

    --help
    -h shows this message
""")
    exit(0)

run_modes = {'luigi_silent': ('',), 'dbg': ('silent', 'bw', 'html')}

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover(utils.TESTS_PATH)
    if len(sys.argv) > 1:
        if sys.argv[1] == '-t': 
            results = {}
            for run_mode in run_modes:
                os.environ['UPSG_RUN_MODE'] = run_mode
                for dbg_output_mode in run_modes[run_mode]:
                    os.environ['UPSG_DEBUG_OUTPUT_MODE'] = dbg_output_mode
                    suite_key = '(run_mode={}, output={})'.format(
                            run_mode, 
                            dbg_output_mode)
                    results[suite_key] = unittest.TextTestRunner().run(suite)
            print('#' * 80)
            print('#{}Thorough mode executive summary{}#'.format(
                ' ' * 23,
                ' ' * 24))
            print('#' * 80)
            for suite_key in results:
                result = results[suite_key]
                print('suite: {}: ran {}, failed {}'.format(
                    suite_key,
                    result.testsRun,
                    len(result.failures)))
                result.printErrors()
                print('#' * 80)
                print()
            total_failures = sum((len(result.failures) for result in results.itervalues()))
            print('Summary:')
            print('Ran {} suites: ran {}, failed {}'.format(
                len(results),
                sum((result.testsRun for result in results.itervalues())),
                total_failures))
            exit(total_failures)        
        usage()    
    os.environ['UPSG_RUN_MODE'] = ''
    os.environ['UPSG_DEBUG_OUTPUT_MODE'] = ''
    result = unittest.TextTestRunner().run(suite)
    exit(len(result.failures))
