"""Isaac-free typed interfaces shared by simulation and MPC processes."""

from .protocol import MPCCommandBatch, MPCOutputBatch, RobotStateBatch

__all__ = ["RobotStateBatch", "MPCCommandBatch", "MPCOutputBatch"]
