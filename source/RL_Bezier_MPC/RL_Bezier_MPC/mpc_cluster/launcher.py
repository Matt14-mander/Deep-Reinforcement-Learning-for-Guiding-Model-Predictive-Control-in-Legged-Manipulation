# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPC cluster launcher: spawns W worker processes attached to an existing
shared-memory namespace (the env side must be running as server first).

Usage (normally auto-started by MPCClusterClient via subprocess, but can be
run manually in a separate terminal for debugging):

    python -m RL_Bezier_MPC.mpc_cluster.launcher \
        --namespace rlbmpc_12345 --num-envs 128 --workers 30 \
        --mpc-cfg /tmp/rlbmpc_12345_cfg.json
"""

import argparse
import json
import multiprocessing as mp
import signal
import sys

from .defs import partition_envs
from .worker import worker_main


def main(argv=None):
    parser = argparse.ArgumentParser(description="RL_Bezier_MPC solver cluster")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--mpc-cfg", required=True, help="Path to mpc cfg JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    with open(args.mpc_cfg) as f:
        mpc_cfg = json.load(f)

    parts = partition_envs(args.num_envs, args.workers)
    print(f"[MPCCluster] namespace={args.namespace} envs={args.num_envs} "
          f"workers={len(parts)} horizon={mpc_cfg['mpc_horizon_steps']}", flush=True)

    # spawn (not fork): each worker gets a clean interpreter; nothing heavy is
    # inherited and crocoddyl is imported fresh per process.
    ctx = mp.get_context("spawn")
    procs = []
    for wid, envs in enumerate(parts):
        p = ctx.Process(
            target=worker_main,
            name=f"mpc-worker-{wid}",
            args=(args.namespace, args.num_envs, wid, envs.start, envs.stop,
                  mpc_cfg, args.verbose),
            daemon=False,
        )
        p.start()
        procs.append(p)

    def _terminate(*_):
        print("[MPCCluster] terminating workers...", flush=True)
        for p in procs:
            p.terminate()
        sys.exit(1)

    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode not in (0, None):
            exit_code = p.exitcode
            print(f"[MPCCluster] {p.name} exited with {p.exitcode}", flush=True)
    print("[MPCCluster] all workers exited.", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
