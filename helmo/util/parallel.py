import sys
import multiprocessing
import queue

import tblib.pickling_support


tblib.pickling_support.install()


def in_separate_process(func):
    def target_function(return_q, err_q, *args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return_q.put(res)
        except Exception as e:
            err_q.put(e)
            exc_info = sys.exc_info()
            err_q.put(exc_info)

    def wrapper(*args, **kwargs):
        return_q = multiprocessing.Queue()
        err_q = multiprocessing.Queue()
        p = multiprocessing.Process(target=target_function, args=tuple([return_q, err_q]+list(args)), kwargs=kwargs)
        p.start()
        err = None
        while True:
            if err is not None:
                sys.excepthook(*exc_info)
                raise err
            try:
                res = return_q.get(block=False)
                break
            except queue.Empty:
                pass
            try:
                err = err_q.get(block=False)
                exc_info = err_q.get()
                continue
            except queue.Empty:
                pass
        p.join()
        return res
    return wrapper
