"""Resource limiting utilities for TTS processing.

This module provides cross-platform functions to restrict CPU, GPU, and memory
usage to prevent the TTS application from overwhelming system resources.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResourceConfig:
    """Configuration for resource limits.
    
    Attributes:
        max_cpu_cores: Maximum number of CPU cores to use. None means no limit.
        max_torch_threads: Maximum PyTorch intra-op threads. None means no limit.
        max_gpu_memory_fraction: Maximum GPU memory fraction (0.0-1.0). None means no limit.
        low_priority: If True, lower the process priority.
    """
    max_cpu_cores: Optional[int] = None
    max_torch_threads: Optional[int] = 4
    max_gpu_memory_fraction: Optional[float] = 0.75
    low_priority: bool = True


def get_cpu_count() -> int:
    """Get the number of available CPU cores."""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def set_cpu_affinity(max_cores: Optional[int] = None) -> bool:
    """Restrict the process to use only specific CPU cores.
    
    Args:
        max_cores: Maximum number of CPU cores to use. If None or >= available,
                   no restriction is applied.
    
    Returns:
        True if affinity was set successfully, False otherwise.
    """
    if max_cores is None:
        return False
    
    total_cores = get_cpu_count()
    if max_cores >= total_cores:
        logger.debug(f"max_cores ({max_cores}) >= available ({total_cores}), skipping affinity.")
        return False
    
    # Try using psutil if available (cross-platform)
    try:
        import psutil
        p = psutil.Process()
        # Use first N cores
        cores_to_use = list(range(min(max_cores, total_cores)))
        p.cpu_affinity(cores_to_use)
        logger.info(f"CPU affinity set to cores: {cores_to_use}")
        return True
    except ImportError:
        logger.debug("psutil not available for CPU affinity.")
    except Exception as e:
        logger.warning(f"Failed to set CPU affinity via psutil: {e}")
    
    # Try using os.sched_setaffinity (Linux only)
    if hasattr(os, 'sched_setaffinity'):
        try:
            cores_to_use = set(range(min(max_cores, total_cores)))
            os.sched_setaffinity(0, cores_to_use)
            logger.info(f"CPU affinity set to cores: {cores_to_use}")
            return True
        except Exception as e:
            logger.warning(f"Failed to set CPU affinity via os.sched_setaffinity: {e}")
    
    logger.debug("CPU affinity setting not supported on this platform.")
    return False


def set_process_priority(low_priority: bool = True) -> bool:
    """Lower the process priority to reduce impact on system responsiveness.
    
    Args:
        low_priority: If True, set the process to lower priority.
    
    Returns:
        True if priority was changed, False otherwise.
    """
    if not low_priority:
        return False
    
    # Try using os.nice (Unix-like systems)
    if hasattr(os, 'nice'):
        try:
            # Nice value 10 is moderately lower priority (range: -20 to 19)
            os.nice(10)
            logger.info("Process priority lowered via os.nice(10)")
            return True
        except Exception as e:
            logger.warning(f"Failed to set nice value: {e}")
    
    # Try using psutil (cross-platform, including Windows)
    try:
        import psutil
        p = psutil.Process()
        if sys.platform == 'win32':
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            logger.info("Process priority set to BELOW_NORMAL (Windows)")
        else:
            p.nice(10)
            logger.info("Process priority lowered via psutil.nice(10)")
        return True
    except ImportError:
        logger.debug("psutil not available for priority setting.")
    except Exception as e:
        logger.warning(f"Failed to set process priority via psutil: {e}")
    
    return False


def set_torch_thread_limits(max_threads: Optional[int] = None) -> bool:
    """Limit the number of threads PyTorch uses for CPU operations.
    
    Args:
        max_threads: Maximum number of threads. If None, no limit is applied.
    
    Returns:
        True if limits were applied, False otherwise.
    """
    if max_threads is None:
        return False
    
    try:
        import torch
        
        # Limit intra-op parallelism (within a single operator)
        torch.set_num_threads(max_threads)
        
        # Limit inter-op parallelism (between operators)
        torch.set_num_interop_threads(max(1, max_threads // 2))
        
        logger.info(f"PyTorch threads limited: num_threads={max_threads}, "
                   f"interop_threads={max(1, max_threads // 2)}")
        return True
    except ImportError:
        logger.debug("PyTorch not available.")
    except Exception as e:
        logger.warning(f"Failed to set PyTorch thread limits: {e}")
    
    return False


def set_gpu_memory_limit(fraction: Optional[float] = None, device: Optional[str] = None) -> bool:
    """Limit GPU memory usage for CUDA devices.
    
    Args:
        fraction: Maximum memory fraction (0.0-1.0). If None, no limit is applied.
        device: The device string (e.g., 'cuda', 'cuda:0'). If None, applies to default.
    
    Returns:
        True if limit was applied, False otherwise.
    """
    if fraction is None:
        return False
    
    # Only applies to CUDA devices
    if device and device not in ('cuda', 'mps') and not device.startswith('cuda:'):
        return False
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.debug("CUDA not available, skipping GPU memory limit.")
            return False
        
        # Clamp fraction to valid range
        fraction = max(0.1, min(1.0, fraction))
        
        # Set per-process memory fraction
        torch.cuda.set_per_process_memory_fraction(fraction)
        
        # Also limit memory fragmentation by setting the max split size
        # This helps with memory efficiency
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            logger.info(f"CUDA memory limited to {fraction * 100:.0f}% of available GPU memory")
            return True
            
    except ImportError:
        logger.debug("PyTorch not available.")
    except Exception as e:
        logger.warning(f"Failed to set GPU memory limit: {e}")
    
    return False


def set_environment_limits(max_threads: Optional[int] = None) -> None:
    """Set environment variables that affect threading before importing heavy libraries.
    
    This should be called as early as possible, ideally before importing torch.
    
    Args:
        max_threads: Maximum number of threads for various libraries.
    """
    if max_threads is None:
        return
    
    thread_str = str(max_threads)
    
    # OpenMP (used by many numerical libraries)
    os.environ.setdefault('OMP_NUM_THREADS', thread_str)
    
    # MKL (Intel Math Kernel Library)
    os.environ.setdefault('MKL_NUM_THREADS', thread_str)
    
    # OpenBLAS
    os.environ.setdefault('OPENBLAS_NUM_THREADS', thread_str)
    
    # NumExpr
    os.environ.setdefault('NUMEXPR_NUM_THREADS', thread_str)
    
    # Limit PyTorch's use of all cores
    os.environ.setdefault('TORCH_NUM_THREADS', thread_str)
    
    logger.debug(f"Environment thread limits set to {max_threads}")


def apply_resource_limits(config: Optional[ResourceConfig] = None, device: Optional[str] = None) -> dict:
    """Apply all resource limits based on configuration.
    
    This is the main entry point for applying resource restrictions. It should
    be called at the start of worker processes.
    
    Args:
        config: ResourceConfig with limit settings. If None, uses defaults.
        device: The compute device being used (for GPU limits).
    
    Returns:
        A dict with keys for each limit type and bool values indicating success.
    """
    if config is None:
        config = ResourceConfig()
    
    results = {
        'cpu_affinity': False,
        'process_priority': False,
        'torch_threads': False,
        'gpu_memory': False,
    }
    
    logger.info(f"Applying resource limits: max_cpu_cores={config.max_cpu_cores}, "
               f"max_torch_threads={config.max_torch_threads}, "
               f"max_gpu_memory={config.max_gpu_memory_fraction}, "
               f"low_priority={config.low_priority}")
    
    # Apply limits
    results['cpu_affinity'] = set_cpu_affinity(config.max_cpu_cores)
    results['process_priority'] = set_process_priority(config.low_priority)
    results['torch_threads'] = set_torch_thread_limits(config.max_torch_threads)
    results['gpu_memory'] = set_gpu_memory_limit(config.max_gpu_memory_fraction, device)
    
    applied = [k for k, v in results.items() if v]
    if applied:
        logger.info(f"Successfully applied resource limits: {', '.join(applied)}")
    else:
        logger.info("No resource limits were applied (platform may not support them).")
    
    return results


def get_memory_info() -> dict:
    """Get current memory usage information.
    
    Returns:
        A dict with memory statistics, or empty dict if unavailable.
    """
    info = {}
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_gb'] = mem.total / (1024 ** 3)
        info['available_gb'] = mem.available / (1024 ** 3)
        info['percent_used'] = mem.percent
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not get memory info: {e}")
    
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Could not get GPU memory info: {e}")
    
    return info
