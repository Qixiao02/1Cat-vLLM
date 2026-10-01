# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-size pinned host memory for the Qwen4Exp PLE host table.

``torch.empty(..., pin_memory=True)`` is served by PyTorch's CUDA caching host
allocator, which rounds every request up to the next power of two and pins the
whole block: an 11.92 GiB host table pins 16 GiB per tensor-parallel rank, and
the four ranks of one host together hold about 16 GiB that nothing reads
(measured: MemAvailable fell by 63-67 GiB while 47.7 GiB were expected).
``vllm/v1/simple_kv_offload/worker.py`` works around the same rounding by
registering ordinary memory with ``cudaHostRegister``.

This module does that for the PLE table:

* an anonymous, private, page-aligned mapping of the table's size (rounded up
  to a whole page, nothing more) holds the rows;
* its pages are faulted in for writing, then page-locked with the driver's
  ``cuMemHostRegister`` (portable + device-mapped, which is what UVA gives
  ``cudaHostAlloc`` memory too), and the result is a CPU tensor over exactly
  the requested bytes that ``Tensor.is_pinned()`` reports as pinned, so the UVA
  view in ``csrc/cuda_view.cu`` takes its ``cudaHostGetDevicePointer`` branch
  and never copies;
* the registration is dropped (``cuMemHostUnregister``) when the last
  reference to the storage goes away, before the mapping is unmapped.

The driver API is used on purpose: a failing runtime ``cudaHostRegister``
leaves its error in the runtime's per-thread "last error" slot, from where the
next ``C10_CUDA_KERNEL_LAUNCH_CHECK`` of an unrelated torch kernel would
report it. A failing ``cuMemHostRegister`` leaves nothing behind.

``SX_OPT_PLE_EXACT_PIN`` (default "1"; "0" restores ``pin_memory=True``)
selects the path. Whenever the exact path cannot produce a pinned table, a
warning is logged once and the caching-allocator allocation is used, so the
table is never left unpinned.
"""

from __future__ import annotations

import math
import mmap
import os
import sys
import weakref

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

EXACT_PIN_ENV = "SX_OPT_PLE_EXACT_PIN"
_OFF_VALUES = frozenset({"0", "false", "off", "no"})

# CUDA driver API cuMemHostRegister flags (cuda.h).
_CU_MEMHOSTREGISTER_PORTABLE = 0x01
_CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
_REGISTER_FLAGS = _CU_MEMHOSTREGISTER_PORTABLE | _CU_MEMHOSTREGISTER_DEVICEMAP

_fallback_warned = False


def exact_pin_enabled() -> bool:
    """SX_OPT_PLE_EXACT_PIN: anything but 0/false/off/no (default "1") is on."""
    return os.environ.get(EXACT_PIN_ENV, "1").strip().lower() not in _OFF_VALUES


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def caching_allocator_bytes(nbytes: int) -> int:
    """What the caching host allocator pins for ``nbytes`` (next power of two)."""
    return 1 << max(nbytes - 1, 0).bit_length()


# --- driver / torch hooks (replaced by the CPU tests) ------------------------


def _device_index() -> int:
    return torch.cuda.current_device()


def _check_driver_result(result: object, operation: str) -> None:
    error = result[0] if isinstance(result, tuple) else result
    if int(getattr(error, "value", error)) != 0:
        raise RuntimeError(f"{operation} failed: {error}")


def _host_register(address: int, nbytes: int, device_index: int) -> None:
    """Page-lock ``[address, address + nbytes)``; raises on any error."""
    from cuda.bindings import driver

    # The driver API needs the device's context to be current on this thread;
    # the guard makes it so (the context exists already: the table's device
    # half is allocated before the host half).
    with torch.cuda.device(device_index):
        _check_driver_result(
            driver.cuMemHostRegister(address, nbytes, _REGISTER_FLAGS),
            "cuMemHostRegister",
        )


def _host_unregister(address: int, device_index: int) -> None:
    from cuda.bindings import driver

    with torch.cuda.device(device_index):
        _check_driver_result(driver.cuMemHostUnregister(address), "cuMemHostUnregister")


def _is_pinned(tensor: torch.Tensor) -> bool:
    return tensor.is_pinned()


# --- the exact-size allocation ----------------------------------------------


def _anonymous_mapping(length: int) -> mmap.mmap:
    """A zero-filled, private, page-aligned mapping of ``length`` bytes."""
    if hasattr(mmap, "MAP_PRIVATE") and hasattr(mmap, "MAP_ANONYMOUS"):
        mapping = mmap.mmap(-1, length, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    else:  # Windows: only used by the CPU tests
        mapping = mmap.mmap(-1, length)
    # Registered pages are pinned, and Linux copies pinned anonymous pages
    # eagerly at fork(): a forked child would duplicate the whole table, which
    # it never reads.
    dont_fork = getattr(mmap, "MADV_DONTFORK", None)
    if dont_fork is not None:
        try:
            mapping.madvise(dont_fork)
        except (OSError, ValueError):
            pass
    return mapping


def _release(
    address: int,
    mapping: mmap.mmap,  # noqa: ARG001 - an argument so the finalizer holds it
    device_index: int,
) -> None:
    """Finalizer: unregister first; ``mapping`` is unmapped after this returns."""
    if sys.is_finalizing():
        return  # the process is going away; the driver reclaims everything
    try:
        _host_unregister(address, device_index)
    except Exception as exc:  # noqa: BLE001 - nothing to do about it in a finalizer
        logger.warning("PLE host table: cuMemHostUnregister failed: %s", exc)


def allocate_exact_pinned(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A pinned, page-aligned CPU tensor of exactly ``shape``/``dtype`` bytes.

    The backing mapping is ``ceil(nbytes / page_size) * page_size`` bytes long
    and registered in full. The tensor keeps everything alive; dropping its
    last reference (and the UVA view's, which holds it) unregisters and unmaps.
    Raises if anything fails; nothing is left registered in that case.
    """
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if nbytes <= 0:
        raise ValueError(f"cannot pin an empty table of shape {tuple(shape)}")
    page = mmap.PAGESIZE
    length = _round_up(nbytes, page)
    device_index = _device_index()

    mapping = _anonymous_mapping(length)
    flat = np.frombuffer(mapping, dtype=np.uint8, count=nbytes)
    address = flat.ctypes.data
    if address % page:
        raise RuntimeError(f"mapping at {address:#x} is not page aligned")
    # Fault every page in for writing before pinning, so the pin can never land
    # on the shared zero page of an untouched private page (a later CPU write
    # would replace that page behind the pin's back).
    flat[::page] = 0

    _host_register(address, length, device_index)
    try:
        table = torch.from_numpy(flat).view(dtype).reshape(shape)
        if not _is_pinned(table):
            raise RuntimeError(
                "cuMemHostRegister succeeded but torch does not report the "
                "memory as pinned"
            )
    except BaseException:
        try:
            _host_unregister(address, device_index)
        except Exception:  # noqa: BLE001 - keep the original error
            logger.warning("PLE host table: cuMemHostUnregister failed", exc_info=True)
        raise
    # torch.from_numpy keeps ``flat`` alive for as long as the storage lives, so
    # this runs once nothing can reach the memory any more. atexit is off: at
    # interpreter exit the driver tears the registration down with the context.
    finalizer = weakref.finalize(flat, _release, address, mapping, device_index)
    finalizer.atexit = False
    return table


def allocate_host_table(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """The host half of the PLE table: pinned, exact-size when enabled.

    An empty table is not pinned (there is nothing to pin). With
    ``SX_OPT_PLE_EXACT_PIN=0``, or when the exact path fails for any reason,
    this is the allocation the table always used:
    ``torch.empty(..., pin_memory=True)``.
    """
    global _fallback_warned
    pin = shape[0] > 0
    if pin and exact_pin_enabled():
        try:
            table = allocate_exact_pinned(shape, dtype)
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            if not _fallback_warned:
                _fallback_warned = True
                logger.warning(
                    "Qwen4Exp PLE host table: exact-size pinning failed (%s: %s); "
                    "falling back to torch.empty(pin_memory=True), which pins up "
                    "to twice as much host memory (power-of-two rounding). Set "
                    "%s=0 to silence this.",
                    type(exc).__name__,
                    exc,
                    EXACT_PIN_ENV,
                )
        else:
            nbytes = table.numel() * table.element_size()
            logger.info(
                "Qwen4Exp PLE host table: %d bytes page-locked at their exact "
                "size (cuMemHostRegister; the caching host allocator would pin "
                "%d)",
                nbytes,
                caching_allocator_bytes(nbytes),
            )
            return table
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin)
