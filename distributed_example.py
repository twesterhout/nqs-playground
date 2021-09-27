import os
import re
import socket
import torch
import torch.distributed
# import torch.multipro

def init_slurm(fn, backend="gloo"):
    slurm_nodelist = os.environ["SLURM_NODELIST"]
    root_node = slurm_nodelist.split(" ")[0].split(",")[0]
    if "[" in root_node:
        name, numbers = root_node.split("[", maxsplit=1)
        number = numbers.split(",", maxsplit=1)[0]
        if "-" in number:
            number = number.split("-")[0]
        number = re.sub("[^0-9]", "", number)
        root_node = name + number
    os.environ["MASTER_ADDR"] = root_node

    port = os.environ["SLURM_JOB_ID"]
    port = port[-4:] # use the last 4 numbers in the job id as the id
    port = int(port) + 15000 # all ports should be in the 10k+ range
    os.environ["MASTER_PORT"] = str(port)

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
    fn()

def _local_init_process(rank, size, fn, backend):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12910"
    torch.distributed.init_process_group(backend, rank=rank, world_size=size)
    fn()

def init_local(size, fn, backend="gloo"):
    import torch.multiprocessing

    processes = []
    torch.multiprocessing.set_start_method("spawn")
    for rank in range(size):
        p = torch.multiprocessing.Process(target=_local_init_process, args=(rank, size, fn, backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def run():
    print(
        "Hello from process {} on node {} out of {}"
        "".format(torch.distributed.get_rank(), socket.gethostname(), torch.distributed.get_world_size())
    )

def main():
    if "SLURM_JOB_ID" in os.environ:
        init_slurm(run)
    else:
        init_local(torch.cuda.device_count(), run)

if __name__ == '__main__':
    main()
