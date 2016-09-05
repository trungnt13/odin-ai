import unittest


def run(device='cpu'):
    tests = [
        'backend_test',
        'nnet_test',
        'compare_test',
        'rnn_test',
        'model_test'
    ]
    import os
    os.environ['ODIN'] = '%s,float64'
    for t in tests:
        print('\n************ Running: %s ************' % t)
        exec('from . import %s' % t)
        tests = unittest.TestLoader().loadTestsFromModule(globals()[t])
        unittest.TextTestRunner(verbosity=2).run(tests)
