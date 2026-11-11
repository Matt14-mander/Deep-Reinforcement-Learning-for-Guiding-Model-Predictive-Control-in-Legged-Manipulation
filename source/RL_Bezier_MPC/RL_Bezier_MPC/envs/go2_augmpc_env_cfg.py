"""Fixed-rate high-level Go2 feedback over the existing 50 Hz Torch MPC."""
import math
from isaaclab.utils import configclass
from .quadruped_mpc_env_cfg import QuadrupedMPCEnvCfg
from .go2_feedback import OBSERVATION_SIZE


@configclass
class Go2AugMPCEnvCfg(QuadrupedMPCEnvCfg):
    high_level_hz: float = 4.
    policy_velocity_source: str = "measured"
    residual_limit_m: float = .001
    goal_tolerance_m: float = .004
    stop_ticks: int = 50
    episode_length_s: float = 20.
    mpc_backend: str = "torch_tracking"
    use_mpc_cluster: bool = False
    torch_foot_plane_height: float = .022
    target_min_m: float = .064
    target_max_m: float = .100

    def __post_init__(self):
        # The base class derives robot/task settings using its legacy integer ratio.
        self.decimation=self.mpc_decimation*self.rl_policy_period
        super().__post_init__()
        if not math.isfinite(self.high_level_hz) or self.high_level_hz<=0:
            raise ValueError("High-level frequency must be finite and positive")
        period=1/self.high_level_hz
        self.decimation=round(period/self.sim.dt)
        if self.decimation<1 or abs(self.decimation*self.sim.dt-period)>1e-9:
            raise ValueError("High-level period must contain an integer number of physics steps")
        if self.policy_velocity_source not in ("measured","mpc_prediction"):
            raise ValueError("Invalid velocity source")
        self.action_space=4
        self.observation_space=OBSERVATION_SIZE
        self.state_space=0
        self.scene.num_envs=8
        self.scene.env_spacing=1.15
        self.sim.render_interval=1000000
