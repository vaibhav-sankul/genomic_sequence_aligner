"""
CUDA Hardware and Runtime Detection Module
Detects NVIDIA GPU hardware, CUDA drivers, Numba CUDA bindings, and device capabilities.
Enables seamless, crash-free automatic fallback to CPU execution.
"""

import sys
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Detailed summary of the host system's compute and GPU capabilities."""
    has_cuda: bool
    device_count: int
    device_name: str
    compute_capability: Optional[str]
    total_memory_mb: float
    free_memory_mb: float
    numba_cuda_available: bool
    driver_version: Optional[str]
    runtime_version: Optional[str]
    active_mode: str  # 'CUDA' or 'CPU_FALLBACK'

    def summary(self) -> Dict[str, Any]:
        return {
            "has_cuda": self.has_cuda,
            "device_count": self.device_count,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "total_memory_mb": self.total_memory_mb,
            "active_mode": self.active_mode
        }


class CUDADetector:
    """
    Probes system hardware for CUDA acceleration capabilities.
    Guarantees no unhandled exceptions even if no NVIDIA GPU or CUDA runtime exists.
    """

    _cached_info: Optional[HardwareInfo] = None

    @classmethod
    def get_hardware_info(cls, force_refresh: bool = False) -> HardwareInfo:
        if cls._cached_info is not None and not force_refresh:
            return cls._cached_info

        has_cuda = False
        device_count = 0
        device_name = "None (CPU Execution)"
        compute_capability = None
        total_mem_mb = 0.0
        free_mem_mb = 0.0
        numba_cuda_avail = False
        driver_ver = None
        runtime_ver = None

        try:
            from numba import cuda
            numba_cuda_avail = cuda.is_available()
            
            if numba_cuda_avail:
                devices = cuda.list_devices()
                device_count = len(devices)
                if device_count > 0:
                    current_dev = cuda.get_current_device()
                    device_name = current_dev.name.decode("utf-8", errors="ignore") if isinstance(current_dev.name, bytes) else str(current_dev.name)
                    compute_capability = f"{current_dev.compute_capability[0]}.{current_dev.compute_capability[1]}"
                    
                    try:
                        ctx = cuda.current_context()
                        free_mem, total_mem = ctx.get_memory_info()
                        total_mem_mb = round(total_mem / (1024 * 1024), 2)
                        free_mem_mb = round(free_mem / (1024 * 1024), 2)
                    except Exception:
                        pass
                    
                    has_cuda = True
        except Exception as e:
            logger.debug(f"CUDA detection encountered non-fatal error: {e}")
            has_cuda = False

        active_mode = "CUDA" if has_cuda else "CPU_FALLBACK"

        info = HardwareInfo(
            has_cuda=has_cuda,
            device_count=device_count,
            device_name=device_name,
            compute_capability=compute_capability,
            total_memory_mb=total_mem_mb,
            free_memory_mb=free_mem_mb,
            numba_cuda_available=numba_cuda_avail,
            driver_version=driver_ver,
            runtime_version=runtime_ver,
            active_mode=active_mode
        )

        cls._cached_info = info
        return info

    @classmethod
    def is_cuda_available(cls) -> bool:
        """Convenience method checking whether CUDA can be used for execution."""
        return cls.get_hardware_info().has_cuda
