"""Run three independent Isaac processes and compare the same first-shift MPC indices."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def compare(runs):
    z, pos, neg = runs
    rows = z["samples"]
    if len(rows) < 10:
        raise RuntimeError("Incomplete first-shift trace")
    indices = [r[0] for r in rows]
    if indices != list(range(indices[0], indices[-1]+1)):
        raise RuntimeError("Non-contiguous phase indices")
    mid = slice(len(rows)//3, 2*len(rows)//3)
    for sign, run in zip((0, 1, -1), runs):
        if max(abs(x-y) for x,y in zip(run["initial"],z["initial"], strict=True)) > 1e-6:
            raise RuntimeError("Initial state mismatch")
        expected = [[0.,0.,0.],[sign*.004,0.,0.],[sign*.004,0.,0.],[0.,0.,0.]]
        if any(abs(x-y)>1e-7 for a,b in zip(run["controls"],expected) for x,y in zip(a,b)):
            raise RuntimeError("Unexpected decoded control points")
        if run["landings"] != 8 or max(run["drift_mm"]) > 10:
            raise RuntimeError("Crawl gate failed")
        trace = run["samples"]
        if [r[0] for r in trace] != indices:
            raise RuntimeError("Unmatched plan indices")
        if max(abs(trace[i][1]-rows[i][1]) for i in (0,-1)) > 1e-5:
            raise RuntimeError("Reference endpoints changed")
        if not sign:
            continue
        pairs = list(zip(trace[mid],rows[mid]))
        ref, obs = [1000*sum(x[c]-y[c] for x,y in pairs)/len(pairs) for c in (1,2)]
        torque = math.sqrt(sum((a-b)**2 for x,y in pairs for a,b in zip(x[3],y[3]))/(12*len(pairs)))
        print(json.dumps(dict(sign=sign, reference_delta_mm=ref, physx_delta_mm=obs,
                              mpc_torque_delta_rms_Nm=torque, landings=run["landings"],
                              max_drift_mm=max(run["drift_mm"]))), flush=True)
        if sign*ref < 1 or sign*obs < .5 or torque < .005:
            raise RuntimeError("Insufficient directional CoM response or torque change")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    runs = []
    with tempfile.TemporaryDirectory() as folder:
        for name, mm in (("zero",0),("positive",4),("negative",-4)):
            path = Path(folder)/f"{name}.json"
            print(f"Starting {name} action trial", flush=True)
            subprocess.run([sys.executable,"-u",str(Path(__file__).with_name("check_go2_torch_crawl.py")),
                            "--headless","--device",args.device,"--support-shift-fraction","0.75",
                            "--probe-mm",str(mm),"--probe-result",str(path)],check=True)
            runs.append(json.loads(path.read_text(encoding="utf-8")))
    compare(runs)
    print("PASS: single Go2 PhysX Bezier action sensitivity", flush=True)


if __name__ == "__main__":
    main()
