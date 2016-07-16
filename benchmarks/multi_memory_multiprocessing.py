# ===========================================================================
# Conclusion: each processes has its own globals()
# ===========================================================================
from multiprocessing import Process


def check(i, n):
    globals()['Process_%d' % i] = i
    print([(('Process_%d' % j) in globals()) for j in range(n)])


def func(i, n):
    check(i, n)

n = 2
p = [Process(target=func, args=(i, n)) for i in range(n)]
[i.start() for i in p]
[i.join() for i in p]
