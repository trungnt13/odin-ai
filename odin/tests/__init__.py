import unittest


def run(backend='theano', device='cpu'):
    tests = [
        'fuel_test',
        'backend_test',
        'nnet_test',
        # 'compare_test',
        # 'rnn_test',
        # 'model_test'
    ]
    print('*NOTE*: some of the tests probably failed on float32 because of '
          'numerical instable, however, they worked on float64.')
    import os
    os.environ['ODIN'] = '%s,%s,float64' % (backend, device)
    for t in tests:
        print('\n************ Running: %s ************' % t)
        exec('from . import %s' % t)
        tests = unittest.TestLoader().loadTestsFromModule(globals()[t])
        unittest.TextTestRunner(verbosity=2).run(tests)
