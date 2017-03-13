# class Child(Parent1, Parent2):
#     def __init__(self):
#         Parent1.__init__(self)
#         Parent2.__init__(self)

from __future__ import print_function, division, absolute_import


class S1(object):

    def __init__(self):
        super(S1, self).__init__()
        print("S1 init")

    def test(self):
        print("S1")


class S2(object):

    def __init__(self):
        super(S2, self).__init__()
        print("S2 init")

    def test(self):
        print("S2")


class S3(S2, S1):

    def __init__(self):
        super(S3, self).__init__()

s = S3() # init order: S1, S2 (reversed of the inheritant order)
s.test() # S2
