import time
from warnings import filterwarnings

import dask
from dask.distributed import Client

from dask_cuda.benchmarks.utils import (
    get_cluster_options,
    parse_benchmark_args,
    setup_memory_pools,
    wait_for_cluster,
)
from dask_cuda.explicit_comms.comms import CommsContext
from dask_cuda.utils import all_to_all


def parse_args():
    special_args = [
        {
            "name": [
                "-t",
                "--type",
            ],
            "choices": ["cpu", "gpu"],
            "default": "gpu",
            "type": str,
            "help": "Do merge with GPU or CPU dataframes",
        },
    ]
    return parse_benchmark_args(
        description="Explicit comms setup", args_list=special_args
    )


def run(client, args):
    start = time.time()
    wait_for_cluster(client, shutdown_on_failure=True)
    end = time.time()
    cluster_bringup = end - start
    start = time.time()
    setup_memory_pools(
        client,
        args.type == "gpu",
        args.rmm_pool_size,
        args.disable_rmm_pool,
        args.rmm_log_directory,
    )
    end = time.time()
    memory_pools = end - start
    start = time.time()
    comms = CommsContext(client)
    end = time.time()
    comms_bringup = end - start
    start = time.time()
    all_to_all(client)
    end = time.time()
    a2a_time = end - start
    del comms

    print(f"Cluster bringup: {cluster_bringup}s")
    print(f"Memory pools: {memory_pools}s")
    print(f"Comms bringup: {comms_bringup}s")
    print(f"Alltoall: {a2a_time}s")
    return cluster_bringup, memory_pools, comms_bringup


def run_client_from_file(args):
    scheduler_file = args.scheduler_file
    if scheduler_file is None:
        raise RuntimeError("Need scheduler file to be provided")
    with Client(scheduler_file=scheduler_file) as client:
        run(client, args)
        client.shutdown()


def run_create_client(args):
    cluster_options = get_cluster_options(args)
    Cluster = cluster_options["class"]
    cluster_args = cluster_options["args"]
    cluster_kwargs = cluster_options["kwargs"]
    scheduler_addr = cluster_options["scheduler_addr"]

    filterwarnings("ignore", message=".*NVLink.*rmm_pool_size.*", category=UserWarning)
    with Cluster(*cluster_args, **cluster_kwargs) as cluster:
        with Client(scheduler_addr if args.multi_node else cluster) as client:
            run(client, args)
            if args.multi_node:
                client.shutdown()


if __name__ == "__main__":
    args = parse_args()
    if args.multiprocessing_method == "forkserver":
        import multiprocessing.forkserver as f

        f.ensure_running()
    with dask.config.set(
        {"distributed.worker.multiprocessing-method": args.multiprocessing_method}
    ):
        if args.scheduler_file is not None:
            run_client_from_file(args)
        else:
            run_create_client(args)
