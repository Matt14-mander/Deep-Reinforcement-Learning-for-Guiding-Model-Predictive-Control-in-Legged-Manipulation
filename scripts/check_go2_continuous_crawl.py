"""Run the distance-driven, 200 mm straight crawl with optional PhysX replay."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from isaaclab.app import AppLauncher


def run(args):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnvCfg
    from RL_Bezier_MPC.envs.stage1b_cycle_env import Stage1BCycleEnv

    cruise = .023
    goal = .200

    def stride(obs):
        remaining = obs[:, :1]
        # Split the last two strides to avoid a residual below the landing envelope.
        return torch.where(remaining <= cruise, remaining, (remaining / 2).clamp_max(cruise))

    class StraightCrawl(Stage1BCycleEnv):
        def _replan_remaining(self):
            x = self.robot.data.root_link_pos_w[0, 0].double() - self.tracking_origins[0, 0]
            remaining = self.goal - (x - self.base_start[0]).item()
            # Keep a successor cycle available until the measured residual fits one stride.
            self.total_cycles = self.landings // 4 + (1 if remaining <= cruise else 2)
            super()._replan_remaining()

        def _solve_mpc_control_step(self):
            super()._solve_mpc_control_step()
            if self.segment is None:
                return
            w, x, y, z = self.robot.data.root_link_quat_w[0].tolist()
            yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            if self.start_yaw is None:
                self.start_yaw = yaw
            angle = yaw - self.start_yaw
            self.yaw_deg = math.degrees(math.atan2(math.sin(angle), math.cos(angle)))
            lateral = self.robot.data.root_link_pos_w[0, 1] - self.tracking_origins[0, 1] - self.base_start[1]
            self.peak_lateral = max(self.peak_lateral, 1000*abs(lateral.item()))
            self.peak_yaw = max(self.peak_yaw, abs(self.yaw_deg))
            if not math.isfinite(self.yaw_deg) or abs(self.yaw_deg) > 5:
                raise RuntimeError(f"Heading deviation {self.yaw_deg} deg at tick {self.tick}")

        def step(self, action):
            result = super().step(action)
            if writer:
                from PIL import Image, ImageDraw
                rgb = self.render()
                # The annotator may need extra renders for its first frame. Wait
                # for RGB without advancing physics.
                for _ in range(8):
                    if rgb.size and rgb.any():
                        break
                    self.sim.render()
                    rgb = self.render()
                if not rgb.size or not rgb.any():
                    raise RuntimeError("Camera returned no image")
                frame = Image.fromarray(rgb).convert("RGB")
                draw = ImageDraw.Draw(frame)
                dx = (self.robot.data.root_link_pos_w[0, 0] - self.tracking_origins[0, 0]
                      - self.base_start[0]).item() if self.segment is not None else 0.
                draw.rectangle((0, 436, 800, 480), fill="#192534")
                draw.text((12, 442), f"Go2  |  PhysX    {self.tick*self.cfg.mpc_dt:.1f} s    "
                          f"{self.previous}    landings {self.landings}    x {dx*1000:.1f} / 200 mm    4x")
                draw.line((12, 467, 712, 467), fill="#9bafc2", width=2)
                draw.line((12, 467, 12+3500*max(dx, 0), 467), fill="#55dbb2", width=4)
                writer.stdin.write(frame.tobytes())
            return result

    cfg = QuadrupedMPCEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 37
    cfg.sim.device = args.device
    cfg.mpc_backend = "torch_tracking"
    cfg.use_mpc_cluster = False
    cfg.torch_foot_plane_height = .022
    cfg.episode_length_s = 210.  # Failure timeout, not a planned stopping time.
    cfg.viewer.eye, cfg.viewer.lookat = (.75, -1.3, .60), (.10, 0, .20)
    cfg.viewer.resolution = (800, 480)
    cfg.sim.render_interval = cfg.decimation
    args.output.mkdir(parents=True, exist_ok=False)
    writer = None
    env = StraightCrawl(cfg=cfg, render_mode="rgb_array" if args.video else None)
    env.start_yaw = None
    env.peak_lateral = env.peak_yaw = 0.
    try:
        if args.video:
            writer = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", "800x480", "-framerate", str(4/env.step_dt),
                "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(args.output/"crawl.mp4")], stdin=subprocess.PIPE)
        with torch.inference_mode():
            m = env.run_crawl(200, cfg.seed, stride, initial_goal=goal, initial_cycles=2, initial_stride=cruise)
        cycles = len(m["cycle_dx_mm"])
        m.update(cycles=cycles, final_yaw_deg=env.yaw_deg, peak_abs_yaw_deg=env.peak_yaw,
                 peak_abs_lateral_mm=env.peak_lateral)
        print(json.dumps(m, allow_nan=False), flush=True)
        (args.output/"measurements.json").write_text(json.dumps(m, indent=2, allow_nan=False))
        if not (cycles > 3 and cycles == len(env.replans) and m["landings"] == 4*cycles
                and m["stopped"] and m["arrived"] and m["stable_ticks"] == 50
                and abs(env.yaw_deg) <= 2 and env.peak_lateral <= 100
                and max(m["cumulative_stance_mm"]) <= 10*(cycles+1)
                and m["stride_commands_mm"][-1] < cruise*1000-.5):
            raise RuntimeError("200 mm crawl failed; see measurements.json")
    finally:
        try:
            if writer:
                writer.stdin.close()
                if writer.wait():
                    raise RuntimeError("Video encoding failed")
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--video", action="store_true", help="Record MP4 using ffmpeg")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.video:
        args.enable_cameras = True
    launcher = AppLauncher(args)
    passed = False
    try:
        run(args)
        passed = True
        print("PASS: single Go2 0.2 m continuous straight crawl", flush=True)
    finally:
        try:
            launcher.app.close()
        finally:
            if not passed:
                raise SystemExit(1)
