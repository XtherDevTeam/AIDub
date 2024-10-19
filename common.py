import hashlib
import multiprocessing
import time
import datetime
import typing
import asyncio
from multiprocessing import cpu_count


def panic(what: str):
    raise RuntimeError(what)


def log(msg: typing.Any):
    print(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S][AIDub] "), msg)


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


processes = []


def cleanup_processes():
    global processes
    while processes:
        p = processes.pop()
        p.join()


def get_free_worker(func, args_list, max_workers=32) -> int:
    """
    Get a free worker to run the given function in parallel. If there are no free workers, create a new one.

    :param func: The function to run in parallel.
    :param args_list: A list of arguments to pass to the function.
    :param max_workers: The maximum number of workers to run in parallel. If None, use the number of CPUs.
    :return: The index of the worker to use.
    """
    global processes
    p = multiprocessing.Process(target=func, args=args_list)
    for i in range(len(processes)):
        if not processes[i].is_alive():
            processes[i] = p
            return i

    # no free worker found, create a new one
    if len(processes) < max_workers:
        processes.append(p)
        return len(processes) - 1
    else:
        i = 0
        processes[i].join()
        processes[i] = p
        return i


def run_in_parallel(func, args_list, max_workers=32):
    """
    Run the given function in parallel using the maximum number of workers. If there are no free workers, create a new one.
    :param func: The function to run in parallel.
    :param args_list: A list of arguments to pass to the function.
    :param max_workers: The maximum number of workers to run in parallel. If None, use the number of CPUs.
    :return: None.
    """
    global processes

    if max_workers is None:
        max_workers = cpu_count()

    worker = get_free_worker(func, args_list, max_workers)
    processes[worker].start()
