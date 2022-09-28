import sys

if sys.platform != "linux":
    raise ImportError("Only Linux is supported by Dask-CUDA at this time")


import dask
import dask.dataframe.core
import dask.dataframe.shuffle

from ._version import get_versions
from .cuda_worker import CUDAWorker
from .explicit_comms.dataframe.shuffle import can_use_ec_shuffle, shuffle as ec_shuffle
from .local_cuda_cluster import LocalCUDACluster
from .proxify_device_objects import proxify_decorator, unproxify_decorator

__version__ = get_versions()["version"]
del get_versions

# Register explicit-comms shuffle
registry = dask.dataframe.shuffle.shuffle_registry
registry.register(max(registry.known_priorities) + 1, can_use_ec_shuffle, ec_shuffle)
del registry

# Monkey patching Dask to make use of proxify and unproxify in compatibility mode
dask.dataframe.shuffle.shuffle_group = proxify_decorator(
    dask.dataframe.shuffle.shuffle_group
)
dask.dataframe.core._concat = unproxify_decorator(dask.dataframe.core._concat)
