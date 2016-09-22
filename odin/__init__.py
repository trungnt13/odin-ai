version = '0.1'

# This SIGNAL can terminate running iterator (or generator),
# however, it cannot terminate them at the beginning,
# so you should not call .send(SIG_TERMINATE_ITERATOR)
# at beginning of iterator
SIG_TERMINATE_ITERATOR = '__signal_terminate_iterator__'

SIG_TRAIN_SAVE = '__signal_train_save__'
SIG_TRAIN_ROLLBACK = '__signal_train_rollback__'
SIG_TRAIN_STOP = '__signal_train_stop__'
