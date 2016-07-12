import unittest


def run():
    tests = ['backend_test', 'nnet_test', 'model_test']
    # tests = ['model_test']
    for t in tests:
        print('\n************ Running: %s ************' % t)
        exec('from . import %s' % t)
        tests = unittest.TestLoader().loadTestsFromModule(globals()[t])
        unittest.TextTestRunner(verbosity=2).run(tests)
