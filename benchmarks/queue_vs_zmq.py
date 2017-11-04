import sys
import zmq
from multiprocessing import Process, Queue
import time
import numpy as np
import marshal
from odin.utils import array2bytes, bytes2array
from odin.utils.mpi import QueueZMQ

NB_MESSAGE = 800000 * 2


X = np.random.randn(80, 120).astype('float32')
print(np.sum(X), np.sum(X**2))

context = zmq.Context(io_threads=1)
# context = zmq.Context.instance()


# ===========================================================================
# zmq
# ===========================================================================
def worker_zmq():
    work_receiver = context.socket(zmq.PULL)
    work_receiver.connect("tcp://127.0.0.1:5557")

    start_time = time.time()
    for task_nbr in range(NB_MESSAGE):
        message = work_receiver.recv()
        message = bytes2array(message)
    end_time = time.time()
    duration = end_time - start_time
    msg_per_sec = NB_MESSAGE / duration
    print("Zmq Duration: %s" % duration, msg_per_sec,
          np.sum(message), np.sum(message**2))
    sys.exit(1)


def main_zmq():
    Process(target=worker_zmq, args=()).start()
    ventilator_send = context.socket(zmq.PUSH)
    ventilator_send.bind("tcp://127.0.0.1:5557")

    for num in range(NB_MESSAGE):
        ventilator_send.send(array2bytes(X))


# ===========================================================================
# Queue
# ===========================================================================
def worker_queue(q):
    start_time = time.time()
    for task_nbr in range(NB_MESSAGE):
        message = q.get()
        message = bytes2array(message)
    end_time = time.time()
    duration = end_time - start_time
    msg_per_sec = NB_MESSAGE / duration
    print("Queue Duration: %s" % duration, msg_per_sec,
          np.sum(message), np.sum(message**2))
    sys.exit(1)


def main_queue():
    send_q = Queue()
    Process(target=worker_queue, args=(send_q,)).start()
    for num in range(NB_MESSAGE):
        send_q.put(array2bytes(X))


# ===========================================================================
# Run the test
# ===========================================================================
if __name__ == "__main__":
    # main_zmq()
    main_queue()
    context.term()
