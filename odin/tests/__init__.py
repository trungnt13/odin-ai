import unittest
# python -c "from odin.tests import run; run('theano')"
# python -c "from odin.tests import run; run('tensorflow')"


def run(backend='tensorflow', device='gpu'):
    tests = [
        'utils_test',
        'fuel_test',
        'backend_test',
        'nnet_test',
        'rnn_test',
        # 'compare_test',
        # 'model_test'
    ]
    print('*NOTE*: some of the tests probably failed on float32 because of '
          'numerical instable, however, they worked on float64.')
    import os
    os.environ['ODIN'] = '%s,%s,float32' % (backend, device)
    for t in tests:
        print('\n************ Running: %s ************' % t)
        exec('from . import %s' % t)
        tests = unittest.TestLoader().loadTestsFromModule(globals()[t])
        unittest.TextTestRunner(verbosity=2).run(tests)
