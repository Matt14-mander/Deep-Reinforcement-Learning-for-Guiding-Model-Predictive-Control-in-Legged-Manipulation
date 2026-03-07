# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait scheduler for generating contact sequences for quadruped locomotion.

This module generates ContactSequence objects for standard quadruped gaits.
It is pure logic with NO Crocoddyl dependency, replacing the sequencing logic
in Crocoddyl's demo createWalkingProblem, createTrottingProblem, etc.

Supported gaits:
- trot: Diagonal pairs alternate (RF+LH, then LF+RH)
- walk: One leg at a time (RH→RF→LH→LF)
- pace: Lateral pairs alternate (RF+RH, then LF+LH)
- bound: Front/hind pairs alternate (LF+RF, then LH+RH)
"""

from typing import Dict, List, Optional

from .contact_sequence import ContactPhase, ContactSequence


class GaitScheduler:
    """Generate contact sequences for standard quadruped gaits.

    Gait definitions (which groups of feet swing together):
        trot:  RF+LH swing together, then LF+RH (diagonal pairs)
        walk:  RH→RF→LH→LF (one leg at a time)
        pace:  RF+RH swing together, then LF+LH (lateral pairs)
        bound: LF+RF swing together, then LH+RH (front/hind pairs)

    Each gait cycle consists of alternating swing groups with optional
    double-support phases in between for stability.

    Attributes:
        FOOT_NAMES: List of valid foot names.
        GAIT_PATTERNS: Dictionary of gait definitions.
    """

    FOOT_NAMES = ["LF", "RF", "LH", "RH"]

    GAIT_PATTERNS: Dict[str, Dict] = {
        "trot": {
            "swing_groups": [["RF", "LH"], ["LF", "RH"]],
            "description": "Diagonal pairs alternate",
        },
        "walk": {
            "swing_groups": [["RH"], ["RF"], ["LH"], ["LF"]],
            "description": "One leg at a time",
        },
        "pace": {
            "swing_groups": [["RF", "RH"], ["LF", "LH"]],
            "description": "Lateral pairs alternate",
        },
        "bound": {
            "swing_groups": [["LF", "RF"], ["LH", "RH"]],
            "description": "Front/hind pairs alternate",
        },
    }

    def __init__(self):
        """Initialize the gait scheduler."""
        pass

    @classmethod
    def get_available_gaits(cls) -> List[str]:
        """Return list of available gait types.

        Returns:
            List of gait names.
        """
        return list(cls.GAIT_PATTERNS.keys())

    @classmethod
    def get_gait_description(cls, gait_type: str) -> str:
        """Get description of a gait type.

        Args:
            gait_type: Name of the gait.

        Returns:
            Description string.
        """
        if gait_type not in cls.GAIT_PATTERNS:
            raise ValueError(f"Unknown gait type: {gait_type}. Available: {cls.get_available_gaits()}")
        return cls.GAIT_PATTERNS[gait_type]["description"]

    def generate(
        self,
        gait_type: str,
        step_duration: float = 0.15,
        support_duration: float = 0.05,
        num_cycles: int = 4,
        first_step_fraction: float = 0.5,
        include_initial_support: bool = True,
        include_final_support: bool = True,
    ) -> ContactSequence:
        """Generate a complete ContactSequence for the requested gait.

        Structure per cycle (e.g., trot):
            [full_support] → [swing RF+LH] → [full_support] → [swing LF+RH]

        The first cycle uses first_step_fraction for shorter initial steps,
        which helps the robot accelerate smoothly from standing.

        Args:
            gait_type: Type of gait ("trot", "walk", "pace", "bound").
            step_duration: Duration of each swing phase in seconds. Default 0.15.
            support_duration: Double-support duration between swings. Default 0.05.
            num_cycles: Number of complete gait cycles. Default 4.
            first_step_fraction: First step is this fraction of full step. Default 0.5.
            include_initial_support: Start with full support phase. Default True.
            include_final_support: End with full support phase. Default True.

        Returns:
            ContactSequence with all phases for the gait.

        Raises:
            ValueError: If gait_type is not recognized.
        """
        if gait_type not in self.GAIT_PATTERNS:
            raise ValueError(
                f"Unknown gait type: {gait_type}. Available: {self.get_available_gaits()}"
            )

        pattern = self.GAIT_PATTERNS[gait_type]
        swing_groups = pattern["swing_groups"]

        phases: List[ContactPhase] = []

        # Optional initial full support phase
        if include_initial_support:
            phases.append(
                ContactPhase(
                    support_feet=self.FOOT_NAMES.copy(),
                    swing_feet=[],
                    duration=support_duration,
                    phase_type="support",
                )
            )

        # Generate cycles
        for cycle_idx in range(num_cycles):
            # Determine step duration for this cycle
            if cycle_idx == 0:
                cycle_step_duration = step_duration * first_step_fraction
            else:
                cycle_step_duration = step_duration

            # Generate phases for each swing group in the pattern
            for group_idx, swing_feet in enumerate(swing_groups):
                # Support phase between swings (except at very start)
                if group_idx > 0 or cycle_idx > 0:
                    phases.append(
                        ContactPhase(
                            support_feet=self.FOOT_NAMES.copy(),
                            swing_feet=[],
                            duration=support_duration,
                            phase_type="support",
                        )
                    )

                # Swing phase
                support_feet = [f for f in self.FOOT_NAMES if f not in swing_feet]
                phases.append(
                    ContactPhase(
                        support_feet=support_feet,
                        swing_feet=swing_feet.copy(),
                        duration=cycle_step_duration,
                        phase_type="swing",
                    )
                )

        # Optional final full support phase
        if include_final_support:
            phases.append(
                ContactPhase(
                    support_feet=self.FOOT_NAMES.copy(),
                    swing_feet=[],
                    duration=support_duration,
                    phase_type="support",
                )
            )

        return ContactSequence(phases=phases)

    def generate_single_step(
        self,
        swing_feet: List[str],
        step_duration: float = 0.15,
        support_before: float = 0.05,
        support_after: float = 0.05,
    ) -> ContactSequence:
        """Generate a sequence for a single step (one swing phase).

        Args:
            swing_feet: List of feet to swing.
            step_duration: Duration of swing phase.
            support_before: Support duration before swing.
            support_after: Support duration after swing.

        Returns:
            ContactSequence for single step.
        """
        support_feet = [f for f in self.FOOT_NAMES if f not in swing_feet]

        phases = []

        if support_before > 0:
            phases.append(
                ContactPhase(
                    support_feet=self.FOOT_NAMES.copy(),
                    swing_feet=[],
                    duration=support_before,
                    phase_type="support",
                )
            )

        phases.append(
            ContactPhase(
                support_feet=support_feet,
                swing_feet=swing_feet.copy(),
                duration=step_duration,
                phase_type="swing",
            )
        )

        if support_after > 0:
            phases.append(
                ContactPhase(
                    support_feet=self.FOOT_NAMES.copy(),
                    swing_feet=[],
                    duration=support_after,
                    phase_type="support",
                )
            )

        return ContactSequence(phases=phases)

    def generate_standing(self, duration: float = 1.0) -> ContactSequence:
        """Generate a standing (all feet in contact) sequence.

        Args:
            duration: Standing duration in seconds.

        Returns:
            ContactSequence with single full-support phase.
        """
        return ContactSequence(
            phases=[
                ContactPhase(
                    support_feet=self.FOOT_NAMES.copy(),
                    swing_feet=[],
                    duration=duration,
                    phase_type="support",
                )
            ]
        )

    def generate_jump(
        self,
        flight_duration: float = 0.2,
        support_before: float = 0.1,
        support_after: float = 0.1,
    ) -> ContactSequence:
        """Generate a jumping sequence (takeoff → flight → landing).

        Args:
            flight_duration: Duration of flight phase.
            support_before: Support duration before takeoff.
            support_after: Support duration after landing.

        Returns:
            ContactSequence for jump.
        """
        phases = []

        if support_before > 0:
            phases.append(
                ContactPhase(
                    support_feet=self.FOOT_NAMES.copy(),
                    swing_feet=[],
                    duration=support_before,
                    phase_type="support",
                )
            )

        phases.append(
            ContactPhase(
                support_feet=[],
                swing_feet=self.FOOT_NAMES.copy(),
                duration=flight_duration,
                phase_type="flight",
            )
        )

        if support_after > 0:
            phases.append(
                ContactPhase(
                    support_feet=self.FOOT_NAMES.copy(),
                    swing_feet=[],
                    duration=support_after,
                    phase_type="support",
                )
            )

        return ContactSequence(phases=phases)

    def generate_from_modulation(
        self,
        gait_type: str,
        step_length_mod: float = 1.0,
        step_height_mod: float = 1.0,
        step_frequency_mod: float = 1.0,
        base_step_duration: float = 0.15,
        base_support_duration: float = 0.05,
        num_cycles: int = 4,
    ) -> ContactSequence:
        """Generate gait with RL-modulated parameters.

        This method allows the RL policy to modulate gait timing through
        the step_frequency parameter.

        Args:
            gait_type: Base gait type.
            step_length_mod: Step length modifier (for FootholdPlanner, not timing).
            step_height_mod: Step height modifier (for BezierFootTrajectory).
            step_frequency_mod: Frequency modifier (affects timing).
            base_step_duration: Base step duration before modulation.
            base_support_duration: Base support duration.
            num_cycles: Number of gait cycles.

        Returns:
            ContactSequence with modulated timing.
        """
        # Frequency modulates the step/support durations inversely
        # Higher frequency = shorter durations
        step_duration = base_step_duration / step_frequency_mod
        support_duration = base_support_duration / step_frequency_mod

        return self.generate(
            gait_type=gait_type,
            step_duration=step_duration,
            support_duration=support_duration,
            num_cycles=num_cycles,
        )

    def get_swing_group_for_gait(self, gait_type: str, group_index: int) -> List[str]:
        """Get the swing feet for a specific group in a gait.

        Args:
            gait_type: Type of gait.
            group_index: Index of the swing group (0-indexed).

        Returns:
            List of feet that swing together in that group.
        """
        if gait_type not in self.GAIT_PATTERNS:
            raise ValueError(f"Unknown gait type: {gait_type}")

        swing_groups = self.GAIT_PATTERNS[gait_type]["swing_groups"]
        if group_index < 0 or group_index >= len(swing_groups):
            raise ValueError(
                f"Group index {group_index} out of range for {gait_type} "
                f"(has {len(swing_groups)} groups)"
            )

        return swing_groups[group_index].copy()
