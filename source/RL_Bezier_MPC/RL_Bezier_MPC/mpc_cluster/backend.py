# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IPC backends for the MPC cluster.

Two interchangeable backends behind the same duck-typed interface:

- ``eigenipc``: EigenIPC named shared memory + Producer/Consumer condition
  variables (Linux only; the production backend for Ubuntu training).
- ``local``: plain numpy arrays + threading primitives, single process.
  Used for protocol unit tests on machines without EigenIPC (e.g. the
  Windows dev box) — workers run as threads sharing the arrays directly.

Interface:
    TensorSet: .buf[name] -> (E, cols) numpy staging array
               .push(name, start, end)  stage -> shared
               .pull(name, start, end)  shared -> stage
               .close()
    Producer:  .trigger(), .wait_ack_from(n, ms) -> bool, .close()
    Consumer:  .wait(ms) -> bool, .ack(), .close()
"""

import threading
from typing import Dict, List

import numpy as np

from .defs import TensorSpec, tensor_specs

try:  # POSIX only — absent on the Windows dev machine
    from EigenIPC.PyEigenIPC import Producer as _EProducer
    from EigenIPC.PyEigenIPC import Consumer as _EConsumer
    from EigenIPC.PyEigenIPC import VLevel
    from EigenIPC.PyEigenIPC import dtype as eigenipc_dtype
    try:
        # EigenIPC 1.0.0 conda package exposes wrappers through PyEigenIPCExt.
        from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
    except ImportError:
        # Keep compatibility with older/source installations.
        from EigenIPC.PyEigenIPC.wrappers.shared_data_view import SharedTWrapper

    EIGENIPC_AVAILABLE = True
except ImportError:
    EIGENIPC_AVAILABLE = False


# =============================================================================
# EigenIPC backend
# =============================================================================

class EigenIPCTensorSet:
    """Server (env) or client (worker) view over the shared tensor table."""

    def __init__(self, namespace: str, num_envs: int, horizon_steps: int,
                 is_server: bool, verbose: bool = False):
        if not EIGENIPC_AVAILABLE:
            raise RuntimeError(
                "EigenIPC is not importable in this environment. "
                "Install it (conda -c AndrePatri eigenipc) or use backend='local'."
            )
        self._wrappers: Dict[str, "SharedTWrapper"] = {}
        self.buf: Dict[str, np.ndarray] = {}
        np_dtypes = {"double": np.float64, "int": np.int32}
        ipc_dtypes = {"double": eigenipc_dtype.Double, "int": eigenipc_dtype.Int}
        for spec in tensor_specs(horizon_steps):
            w = SharedTWrapper(
                namespace=namespace,
                basename=spec.basename,
                is_server=is_server,
                n_rows=num_envs,
                n_cols=spec.n_cols,
                verbose=verbose,
                vlevel=VLevel.V1 if verbose else VLevel.V0,
                dtype=ipc_dtypes[spec.dtype],
                fill_value=0,
                safe=True,
                force_reconnection=is_server,
            )
            w.run()
            self._wrappers[spec.basename] = w
            self.buf[spec.basename] = np.zeros(
                (num_envs, spec.n_cols), dtype=np_dtypes[spec.dtype]
            )

    def push(self, name: str, start: int, end: int):
        self._wrappers[name].write_retry(self.buf[name][start:end], start, 0)

    def pull(self, name: str, start: int, end: int):
        self._wrappers[name].read_retry(start, 0, data=self.buf[name][start:end])

    def close(self):
        for w in self._wrappers.values():
            try:
                w.close()
            except Exception:
                pass


class EigenIPCProducer:
    def __init__(self, namespace: str, basename: str, verbose: bool = False):
        self._p = _EProducer(basename, namespace, verbose,
                             VLevel.V1 if verbose else VLevel.V0, True)
        self._p.run()

    def trigger(self):
        self._p.trigger()

    def wait_ack_from(self, n_consumers: int, ms_timeout: int = -1) -> bool:
        return self._p.wait_ack_from(n_consumers, ms_timeout)

    def close(self):
        self._p.close()


class EigenIPCConsumer:
    def __init__(self, namespace: str, basename: str, verbose: bool = False):
        self._c = _EConsumer(basename, namespace, verbose,
                             VLevel.V1 if verbose else VLevel.V0)
        self._c.run()

    def wait(self, ms_timeout: int = -1) -> bool:
        return self._c.wait(ms_timeout)

    def ack(self):
        self._c.ack()

    def close(self):
        self._c.close()


# =============================================================================
# Local (in-process, thread-based) backend for protocol tests
# =============================================================================

class LocalHub:
    """Shared state for one local cluster: arrays + go/ack synchronization."""

    def __init__(self, num_envs: int, horizon_steps: int, num_workers: int):
        np_dtypes = {"double": np.float64, "int": np.int32}
        self.arrays: Dict[str, np.ndarray] = {
            s.basename: np.zeros((num_envs, s.n_cols), dtype=np_dtypes[s.dtype])
            for s in tensor_specs(horizon_steps)
        }
        self.num_workers = num_workers
        self._go_events: List[threading.Event] = [threading.Event() for _ in range(num_workers)]
        self._ack_count = 0
        self._ack_cv = threading.Condition()
        self._next_consumer_id = 0

    # producer side
    def trigger(self):
        with self._ack_cv:
            self._ack_count = 0
        for ev in self._go_events:
            ev.set()

    def wait_ack_from(self, n: int, ms_timeout: int = -1) -> bool:
        deadline = None if ms_timeout < 0 else ms_timeout / 1000.0
        with self._ack_cv:
            return self._ack_cv.wait_for(lambda: self._ack_count >= n, timeout=deadline)

    # consumer side
    def register_consumer(self) -> int:
        cid = self._next_consumer_id
        self._next_consumer_id += 1
        return cid

    def consumer_wait(self, cid: int, ms_timeout: int = -1) -> bool:
        timeout = None if ms_timeout < 0 else ms_timeout / 1000.0
        ok = self._go_events[cid].wait(timeout)
        if ok:
            self._go_events[cid].clear()
        return ok

    def consumer_ack(self):
        with self._ack_cv:
            self._ack_count += 1
            self._ack_cv.notify_all()


class LocalTensorSet:
    """Both 'sides' share the same LocalHub arrays; push/pull are no-ops
    because .buf *is* the shared array (single-process test backend)."""

    def __init__(self, hub: LocalHub):
        self.buf = hub.arrays

    def push(self, name: str, start: int, end: int):
        pass

    def pull(self, name: str, start: int, end: int):
        pass

    def close(self):
        pass


class LocalProducer:
    def __init__(self, hub: LocalHub):
        self._hub = hub

    def trigger(self):
        self._hub.trigger()

    def wait_ack_from(self, n: int, ms_timeout: int = -1) -> bool:
        return self._hub.wait_ack_from(n, ms_timeout)

    def close(self):
        pass


class LocalConsumer:
    def __init__(self, hub: LocalHub):
        self._hub = hub
        self._cid = hub.register_consumer()

    def wait(self, ms_timeout: int = -1) -> bool:
        return self._hub.consumer_wait(self._cid, ms_timeout)

    def ack(self):
        self._hub.consumer_ack()

    def close(self):
        pass
