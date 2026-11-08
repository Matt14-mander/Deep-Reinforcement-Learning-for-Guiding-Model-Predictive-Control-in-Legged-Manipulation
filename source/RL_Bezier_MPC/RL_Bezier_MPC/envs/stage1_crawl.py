"""One trajectory decision per full two-cycle single-Go2 episode."""
import torch

from .fixed_crawl_env import CrawlFailure, FixedCrawlEnv


class Stage1CrawlTask:
    targets_mm = (32, 40, 48)

    def __init__(self, cfg):
        cfg.scene.num_envs = 1
        cfg.mpc_backend = "torch_tracking"
        cfg.use_mpc_cluster = False
        cfg.torch_foot_plane_height = .022
        cfg.episode_length_s = 35.
        cfg.target_threshold = .004
        cfg.target_pos_range = (.032, .048, 0., 0., 0., 0.)
        self.env = FixedCrawlEnv(cfg=cfg)
        self.rng = torch.Generator().manual_seed(cfg.seed)
        self.ready = False

    def reset(self, target_mm=None, seed=None):
        if target_mm is None:
            index = torch.randint(3, (), generator=self.rng).item()
            target_mm = self.targets_mm[index]
        if target_mm not in self.targets_mm:
            raise ValueError("Unsupported Stage 1 distance")
        env = self.env
        env.walking = False
        env.reset(seed=seed)
        env.begin_crawl(.75, endpoint=True, shape=True)
        self.target_mm = target_mm
        env.target_positions.copy_(env.robot.data.root_link_pos_w)
        env.target_positions[:, 0] += target_mm * .001
        self.ready = True
        # Fixed initial pose and gait. Only the relative forward goal varies.
        return torch.tensor([(target_mm - 40.) / 8.], device=env.device)

    @torch.no_grad()
    def step(self, action):
        if not self.ready:
            raise RuntimeError("Reset before issuing another trajectory decision")
        if action.shape != (2,) or not action.isfinite().all() or (action.abs() > 1).any():
            raise ValueError("Expected two finite normalized trajectory controls")
        self.ready = False
        env = self.env
        cmd = action.new_zeros((1, 12))
        width = (env.action_high - env.action_low).reshape(-1)
        cmd[:, [3, 6]] = 2 * (.004 * action[0]) / width[[3, 6]]
        endpoint = .040 + .008 * action[1]
        cmd[:, 9] = 2 * endpoint / width[9]
        try:
            while not env.stopped:
                _, _, term, trunc, _ = env.step(cmd)
                if (term | trunc).any():
                    raise CrawlFailure("Episode terminated before settling")
            pos = env.robot.data.root_link_pos_w[0].double() - env.tracking_origins[0]
            delta = pos - env.base_start
            dx, dy = (delta[:2] * 1000).tolist()
            if not delta.isfinite().all() or abs(env.goal - endpoint.item()) > 1e-6:
                raise RuntimeError("Invalid final measurement or action decoding")
            error = abs(dx - self.target_mm)
            xs, speed, spin, joints = zip(*env.stop_window)
            result = dict(target_mm=self.target_mm, command_mm=1000*env.goal,
                          dx_mm=dx, dy_mm=dy, error_mm=error, landings=env.landings,
                          drift_mm=(1000*env.max_drift).tolist(), stable_ticks=len(xs),
                          stop_span_mm=1000*(max(xs)-min(xs)), max_speed_mps=max(speed),
                          max_spin_rad_s=max(spin), max_joint_rad_s=max(joints))
            if (env.landings != 8 or len(xs) != 50 or abs(dy) > 10
                    or max(result["drift_mm"]) > 10 or abs(dx - 1000*env.goal) > 4):
                raise CrawlFailure(f"Invalid final crawl: {result}")
            result["success"] = error <= env.cfg.target_threshold * 1000
            reward = -error / 8. + float(result["success"])
        except CrawlFailure as exc:
            result = dict(target_mm=self.target_mm, tick=env.tick, landings=env.landings,
                          failure=str(exc), success=False)
            reward = -5.
        return reward, result

    def close(self):
        self.env.close()
