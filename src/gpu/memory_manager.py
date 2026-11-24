"""
GPU Device Memory Manager & Buffer Pooling
Provides memory pooling, pinned host allocation, and PCIe transfer optimization.
"""

import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    from numba import cuda
    _CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    cuda = None
    _CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeviceMemoryManager:
    """
    Manages GPU device buffers, host pinned memory, and device reuse pools.
    Minimizes PCIe allocation overhead by caching reusable device arrays.
    """

    def __init__(self, use_pinned_memory: bool = True):
        self.use_pinned = use_pinned_memory and _CUDA_AVAILABLE
        self._device_buffer_cache: Dict[str, Any] = {}
        self.allocated_bytes = 0
        self.transfer_count = 0

    def allocate_pinned_host(self, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """
        Allocates page-locked (pinned) host memory for accelerated DMA transfers.
        Falls back to standard NumPy array if CUDA is unavailable.
        """
        if self.use_pinned and _CUDA_AVAILABLE:
            try:
                return cuda.pinned_array(shape, dtype=dtype)
            except Exception as e:
                logger.debug(f"Pinned memory allocation failed ({e}), falling back to standard numpy.")
        return np.zeros(shape, dtype=dtype)

    def get_device_buffer(self, name: str, shape: Tuple[int, ...], dtype: np.dtype) -> Any:
        """
        Retrieves or allocates a reusable device array of the specified shape.
        Avoids repetitive cudaMalloc / cudaFree calls inside chunk processing loops.
        """
        if not _CUDA_AVAILABLE:
            return None

        key = f"{name}_{shape}_{dtype}"
        if key in self._device_buffer_cache:
            return self._device_buffer_cache[key]

        # Allocate new device buffer
        dev_arr = cuda.device_array(shape, dtype=dtype)
        self._device_buffer_cache[key] = dev_arr
        self.allocated_bytes += dev_arr.nbytes
        return dev_arr

    def host_to_device(self, host_arr: np.ndarray, stream=None) -> Any:
        """
        Transfers host data to device with optional CUDA stream.
        """
        if not _CUDA_AVAILABLE:
            return host_arr

        self.transfer_count += 1
        return cuda.to_device(host_arr, stream=stream)

    def device_to_host(self, dev_arr: Any, host_arr: Optional[np.ndarray] = None, stream=None) -> np.ndarray:
        """
        Transfers device data back to host.
        """
        if not _CUDA_AVAILABLE or not hasattr(dev_arr, "copy_to_host"):
            return dev_arr

        self.transfer_count += 1
        if host_arr is not None:
            dev_arr.copy_to_host(host_arr, stream=stream)
            return host_arr
        return dev_arr.copy_to_host(stream=stream)

    def synchronize(self) -> None:
        """Ensures all pending device operations have finished."""
        if _CUDA_AVAILABLE:
            cuda.synchronize()

    def clear_pool(self) -> None:
        """Releases cached device buffers and frees device memory."""
        self._device_buffer_cache.clear()
        self.allocated_bytes = 0
        if _CUDA_AVAILABLE:
            try:
                ctx = cuda.current_context()
                ctx.memory_manager.deallocate()
            except Exception:
                pass
