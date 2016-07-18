version = '0.1'

# This SIGNAL can terminate running iterator (or generator),
# however, it cannot terminate them at the beginning,
# so you should not call .send(SIG_ITERATOR_TERMINATE)
# at beginning of iterator
SIG_ITERATOR_TERMINATE = '__signal_terminate_iterator__'
