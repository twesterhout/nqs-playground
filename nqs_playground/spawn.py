# This is basically a copy-paste from torch.multiprocessing.spawn
# except that we use fork rather than spawn.

import torch.multiprocessing as multiprocessing
from torch.multiprocessing.spawn import _wrap, _python_version_check


def spawn(fn, args=(), nprocs=1, join=True):
    _python_version_check()
    mp = multiprocessing.get_context("fork")
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = mp.SimpleQueue()
        process = mp.Process(
            target=_wrap, args=(fn, i, args, error_queue), daemon=False
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    spawn_context = multiprocessing.SpawnContext(processes, error_queues)
    if not join:
        return spawn_context

    # Loop on join until it returns True or raises an exception.
    while not spawn_context.join():
        pass
