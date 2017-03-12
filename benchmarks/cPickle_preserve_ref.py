from __future__ import print_function, division, absolute_import

import cPickle


class Shit1(object):
    """docstring for Shit1"""

    def __init__(self):
        super(Shit1, self).__init__()


class Shit2(object):
    """docstring for Shit2"""

    def __init__(self):
        super(Shit2, self).__init__()

# ===========================================================================
# This will preserve the reference
# ===========================================================================
if False:
    s1 = Shit1()
    s2 = Shit2()
    s2.shit = s1
    print(s1, s2, s2.shit, s1 == s2.shit)
    cPickle.dump((s1, s2), open('/tmp/s1', 'w'))
else:
    s1, s2 = cPickle.load(open('/tmp/s1', 'rb'))
    print(s1, s2, s2.shit, s1 == s2.shit) # True

# ===========================================================================
# This will remove the reference
# ===========================================================================
if False:
    s1 = Shit1()
    s2 = Shit2()
    s2.shit = s1
    print(s1, s2, s2.shit, s1 == s2.shit)
    cPickle.dump(s1, open('/tmp/s1', 'w'))
    cPickle.dump(s2, open('/tmp/s2', 'w'))
else:
    s1 = cPickle.load(open('/tmp/s1', 'rb'))
    s2 = cPickle.load(open('/tmp/s2', 'rb'))
    print(s1, s2, s2.shit, s1 == s2.shit) # False
