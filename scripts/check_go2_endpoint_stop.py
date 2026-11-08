"""Compare three decoded endpoint actions with independent, seeded PhysX episodes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    initial = None
    previous = None
    with tempfile.TemporaryDirectory() as folder:
        for mm in (32, 40, 48):
            path = Path(folder)/f"{mm}.json"
            print(f"Starting endpoint trial: {mm} mm", flush=True)
            subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("check_go2_torch_crawl.py")),
                            "--headless", "--device", args.device, "--support-shift-fraction", "0.75",
                            "--endpoint-result", str(path), "--target-mm", str(mm)], check=True)
            r = json.loads(path.read_text(encoding="utf8"))
            state = r.pop("initial")
            if initial is None:
                initial = state
            if max(abs(a-b) for a,b in zip(state,initial,strict=True)) > 1e-6:
                raise RuntimeError("Trial initial states differ")
            dx = r["dx_m"]
            print(json.dumps(r), flush=True)
            if (abs(dx-mm*.001) > .004 or r["stable_ticks"] != 50 or r["landings"] != 8
                    or max(r["drift_mm"]) > 10 or (previous is not None and dx-previous < .004)):
                raise RuntimeError("Endpoint accuracy, stop, gait or distance separation failed")
            previous = dx
    print("PASS: single Go2 PhysX endpoint distance and stop", flush=True)


if __name__ == "__main__":
    main()
