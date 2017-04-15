from __future__ import print_function

import sys
import timeit
from contextlib import contextmanager
from collections import defaultdict, OrderedDict


@contextmanager
def UnitTimer(factor=1, name=None):
    start = timeit.default_timer()
    yield None
    end = timeit.default_timer()
    # set name for timing task
    name = ''
    if name is not None:
        name = 'Task:%s ' % str(name)
    print('%sTime: %f (sec)' % (name, (end - start) / factor))


class Profile(object):
    """A profile of hierarchical timers.

    Keeps track of timings performed with :class:`Timer`. It also keeps
    track of the way these timings were nested and makes use of this
    information when reporting.

    """

    def __init__(self):
        self.total = defaultdict(int)
        self.current = []
        self.order = OrderedDict()
        self._id = 1

    def __enter__(self):
        self.current.append('ID%d' % self._id)
        self._id += 1
        # We record the order in which sections were first called
        self.order[tuple(self.current)] = None
        self._start = timeit.default_timer()

    def __exit__(self, *args):
        self.total[tuple(self.current)] += timeit.default_timer() - self._start
        self.current.pop()

    def report(self, f=sys.stderr):
        """Print a report of timing information to standard output.

        Parameters
        ----------
        f : object, optional
            An object with a ``write`` method that accepts string inputs.
            Can be a file object, ``sys.stdout``, etc. Defaults to
            ``sys.stderr``.

        """
        total = sum(v for k, v in self.total.items() if len(k) == 1)

        def print_report(keys, level=0):
            subtotal = 0
            for key in keys:
                if len(key) > level + 1:
                    continue
                subtotal += self.total[key]
                section = ' '.join(key[-1].split('_'))
                section = section[0].upper() + section[1:]
                print('{:4}{:15.6f}{:15.2%}'.format(
                    level * '  ' + section, self.total[key],
                    self.total[key] / total
                ), file=f)
                children = [k for k in keys
                            if k[level] == key[level] and
                            len(k) > level + 1]
                child_total = print_report(children, level + 1)
                if children:
                    print('{:4}{:15.6f}{:15.2%}'.format(
                        (level + 1) * '  ' + 'Other',
                        self.total[key] - child_total,
                        (self.total[key] - child_total) / total
                    ), file=f)
            return subtotal

        print('{:4}{:>15}{:>15}'.format('ID', 'Time', '% of total'),
              file=f)
        print('-' * 34, file=f)
        if total:
            print_report(self.order.keys())
        else:
            print('No profile information collected.', file=f)
