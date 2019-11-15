import os

from dask.distributed import Nanny
from distributed.system import MEMORY_LIMIT

from .local_cuda_cluster import cuda_visible_devices
from .utils import CPUAffinity, get_cpu_affinity, get_gpu_count, get_preload_options


def worker_spec(
    interface=None,
    protocol=None,
    dashboard_address=":8787",
    threads_per_worker=1,
    silence_logs=True,
    CUDA_VISIBLE_DEVICES=None,
    enable_tcp_over_ucx=False,
    enable_infiniband=False,
    ucx_net_devices="",
    enable_nvlink=False,
    **kwargs
):
    """ Create a Spec for a CUDA worker.

    The Spec created by this function can be used as a recipe for CUDA workers
    that can be passed to a SpecCluster.

    Parameters
    ----------
    interface: str
        The external interface used to connect to the scheduler.
    protocol: str
        The protocol to used for data transfer, e.g., "tcp" or "ucx".
    dashboard_address: str
        The address for the scheduler dashboard.  Defaults to ":8787".
    threads_per_worker: int
        Number of threads to be used for each CUDA worker process.
    silence_logs: bool
        Disable logging for all worker processes
    CUDA_VISIBLE_DEVICES: str
        String like ``"0,1,2,3"`` or ``[0, 1, 2, 3]`` to restrict activity to
        different GPUs
    enable_tcp_over_ucx: bool
        Set environment variables to enable TCP over UCX, even if InfiniBand
        and NVLink are not supported or disabled.
    enable_infiniband: bool
        Set environment variables to enable UCX InfiniBand support. Implies
        enable_tcp_over_ucx=True.
    ucx_net_device: str or callable
        A string with the interface name to be used for all devices (empty
        string means use default), or a callable function taking an integer
        identifying the GPU index.
    enable_nvlink: bool
        Set environment variables to enable UCX NVLink support. Implies
        enable_tcp_over_ucx=True.

    Examples
    --------
    >>> from dask_cuda.worker_spec import worker_spec
    >>> worker_spec(interface="enp1s0f0", CUDA_VISIBLE_DEVICES=[0, 2], ucx_net_devices=lambda i: "mlx5_%d:1" % (i //2))
    {0: {'cls': distributed.nanny.Nanny,
      'options': {'env': {'CUDA_VISIBLE_DEVICES': '0,2'},
       'interface': 'enp1s0f0',
       'protocol': None,
       'nthreads': 1,
       'data': dict,
       'dashboard_address': ':8787',
       'plugins': [<dask_cuda.utils.CPUAffinity at 0x7fbb8748a860>],
       'silence_logs': True,
       'memory_limit': 135263611392.0,
       'preload': ['dask_cuda.initialize'],
       'preload_argv': ['--create-cuda-context']}},
     2: {'cls': distributed.nanny.Nanny,
      'options': {'env': {'CUDA_VISIBLE_DEVICES': '2,0'},
       'interface': 'enp1s0f0',
       'protocol': None,
       'nthreads': 1,
       'data': dict,
       'dashboard_address': ':8787',
       'plugins': [<dask_cuda.utils.CPUAffinity at 0x7fbb8748a0f0>],
       'silence_logs': True,
       'memory_limit': 135263611392.0,
       'preload': ['dask_cuda.initialize'],
       'preload_argv': ['--create-cuda-context']}}}

    """
    if CUDA_VISIBLE_DEVICES is None:
        CUDA_VISIBLE_DEVICES = os.environ.get(
            "CUDA_VISIBLE_DEVICES", list(range(get_gpu_count()))
        )
    if isinstance(CUDA_VISIBLE_DEVICES, str):
        CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES.split(",")
    CUDA_VISIBLE_DEVICES = list(map(int, CUDA_VISIBLE_DEVICES))
    memory_limit = MEMORY_LIMIT / get_gpu_count()

    spec = {
        i: {
            "cls": Nanny,
            "options": {
                "env": {
                    "CUDA_VISIBLE_DEVICES": cuda_visible_devices(
                        ii, CUDA_VISIBLE_DEVICES
                    )
                },
                "interface": interface,
                "protocol": protocol,
                "nthreads": threads_per_worker,
                "data": dict,
                "dashboard_address": dashboard_address,
                "plugins": [CPUAffinity(get_cpu_affinity(i))],
                "silence_logs": silence_logs,
                "memory_limit": memory_limit,
                **get_preload_options(
                    protocol=protocol,
                    create_cuda_context=True,
                    enable_tcp_over_ucx=enable_tcp_over_ucx,
                    enable_infiniband=enable_infiniband,
                    enable_nvlink=enable_nvlink,
                    ucx_net_devices=ucx_net_devices,
                    cuda_device_index=i,
                ),
            },
        }
        for ii, i in enumerate(CUDA_VISIBLE_DEVICES)
    }

    return spec