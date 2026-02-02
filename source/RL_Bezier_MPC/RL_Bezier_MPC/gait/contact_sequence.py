# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact sequence data structures for quadruped gait management.

This module defines pure data structures describing contact timing and phases.
It has no dependency on Crocoddyl or IsaacLab, making it easy to test and use
standalone.

The key data structures are:
- ContactPhase: Describes a single phase (which feet are in contact, duration)
- ContactSequence: A sequence of ContactPhases forming a complete gait cycle
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ContactPhase:
    """Describes a single contact phase in a gait sequence.

    A contact phase defines which feet are on the ground (support) and which
    are in the air (swing) for a specific duration.

    Attributes:
        support_feet: List of feet on ground, e.g., ["LF", "RH"].
        swing_feet: List of feet in air, e.g., ["RF", "LH"].
        duration: Phase duration in seconds.
        phase_type: Type of phase - one of:
            - "support": All four feet on ground (double/quad support)
            - "swing": Some feet swinging (normal locomotion)
            - "flight": All feet in air (running/jumping)
            - "impulse": Instantaneous contact change (for MPC node type)
    """

    support_feet: List[str]
    swing_feet: List[str]
    duration: float
    phase_type: str = "swing"

    def __post_init__(self):
        """Validate the contact phase configuration."""
        valid_types = {"support", "swing", "flight", "impulse"}
        if self.phase_type not in valid_types:
            raise ValueError(f"phase_type must be one of {valid_types}, got {self.phase_type}")

        if self.duration < 0:
            raise ValueError(f"duration must be non-negative, got {self.duration}")

        # Validate foot names
        all_feet = set(self.support_feet) | set(self.swing_feet)
        valid_feet = {"LF", "RF", "LH", "RH"}
        invalid = all_feet - valid_feet
        if invalid:
            raise ValueError(f"Invalid foot names: {invalid}. Valid names: {valid_feet}")

    @property
    def num_support_feet(self) -> int:
        """Return number of feet in contact."""
        return len(self.support_feet)

    @property
    def num_swing_feet(self) -> int:
        """Return number of feet swinging."""
        return len(self.swing_feet)

    @property
    def is_full_support(self) -> bool:
        """Check if all four feet are in contact."""
        return self.num_support_feet == 4

    @property
    def is_flight(self) -> bool:
        """Check if no feet are in contact (flight phase)."""
        return self.num_support_feet == 0

    def is_foot_in_contact(self, foot_name: str) -> bool:
        """Check if a specific foot is in contact.

        Args:
            foot_name: Foot identifier ("LF", "RF", "LH", or "RH").

        Returns:
            True if foot is in contact (support), False if swinging.
        """
        return foot_name in self.support_feet

    def copy(self) -> "ContactPhase":
        """Create a copy of this contact phase."""
        return ContactPhase(
            support_feet=self.support_feet.copy(),
            swing_feet=self.swing_feet.copy(),
            duration=self.duration,
            phase_type=self.phase_type,
        )


@dataclass
class ContactSequence:
    """A sequence of contact phases forming a complete gait pattern.

    The contact sequence defines the full timing structure of a gait,
    including which feet are in contact during each phase and for how long.

    Attributes:
        phases: List of ContactPhase objects in temporal order.
    """

    phases: List[ContactPhase] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        """Return total duration of all phases in seconds."""
        return sum(p.duration for p in self.phases)

    @property
    def num_phases(self) -> int:
        """Return number of phases in the sequence."""
        return len(self.phases)

    def get_phase_at_time(self, t: float) -> Optional[ContactPhase]:
        """Return the contact phase active at time t.

        Args:
            t: Time in seconds from start of sequence.

        Returns:
            ContactPhase active at time t, or None if t is beyond sequence.
        """
        if t < 0:
            return None

        cumulative_time = 0.0
        for phase in self.phases:
            cumulative_time += phase.duration
            if t < cumulative_time:
                return phase

        # t is beyond the sequence
        return None

    def get_phase_index_at_time(self, t: float) -> int:
        """Return the index of the phase active at time t.

        Args:
            t: Time in seconds from start of sequence.

        Returns:
            Phase index (0-indexed), or -1 if t is invalid.
        """
        if t < 0:
            return -1

        cumulative_time = 0.0
        for i, phase in enumerate(self.phases):
            cumulative_time += phase.duration
            if t < cumulative_time:
                return i

        return -1

    def get_phase_start_time(self, phase_index: int) -> float:
        """Return the start time of a phase by index.

        Args:
            phase_index: Index of the phase.

        Returns:
            Start time in seconds, or -1 if invalid index.
        """
        if phase_index < 0 or phase_index >= len(self.phases):
            return -1.0

        return sum(p.duration for p in self.phases[:phase_index])

    def get_swing_timings(self, foot_name: str) -> List[Tuple[float, float]]:
        """Return list of (start_time, end_time) when foot is swinging.

        Args:
            foot_name: Foot identifier ("LF", "RF", "LH", or "RH").

        Returns:
            List of (start, end) time tuples for each swing phase of this foot.
        """
        timings = []
        cumulative_time = 0.0

        for phase in self.phases:
            phase_start = cumulative_time
            phase_end = cumulative_time + phase.duration

            if foot_name in phase.swing_feet:
                timings.append((phase_start, phase_end))

            cumulative_time = phase_end

        return timings

    def get_contact_timings(self, foot_name: str) -> List[Tuple[float, float]]:
        """Return list of (start_time, end_time) when foot is in contact.

        Args:
            foot_name: Foot identifier ("LF", "RF", "LH", or "RH").

        Returns:
            List of (start, end) time tuples for each contact phase of this foot.
        """
        timings = []
        cumulative_time = 0.0

        for phase in self.phases:
            phase_start = cumulative_time
            phase_end = cumulative_time + phase.duration

            if foot_name in phase.support_feet:
                timings.append((phase_start, phase_end))

            cumulative_time = phase_end

        return timings

    def get_touchdown_times(self, foot_name: str) -> List[float]:
        """Return list of times when foot touches down (swing → contact).

        Args:
            foot_name: Foot identifier.

        Returns:
            List of touchdown times in seconds.
        """
        touchdowns = []
        cumulative_time = 0.0
        prev_swinging = False

        for phase in self.phases:
            is_swinging = foot_name in phase.swing_feet

            # Touchdown occurs at start of a contact phase following a swing phase
            if prev_swinging and not is_swinging:
                touchdowns.append(cumulative_time)

            prev_swinging = is_swinging
            cumulative_time += phase.duration

        return touchdowns

    def get_liftoff_times(self, foot_name: str) -> List[float]:
        """Return list of times when foot lifts off (contact → swing).

        Args:
            foot_name: Foot identifier.

        Returns:
            List of liftoff times in seconds.
        """
        liftoffs = []
        cumulative_time = 0.0
        prev_swinging = True  # Assume starts in swing to catch initial liftoff

        for i, phase in enumerate(self.phases):
            is_swinging = foot_name in phase.swing_feet

            # Liftoff occurs at start of a swing phase following a contact phase
            if not prev_swinging and is_swinging:
                liftoffs.append(cumulative_time)

            prev_swinging = is_swinging
            cumulative_time += phase.duration

        return liftoffs

    def append_phase(self, phase: ContactPhase):
        """Add a phase to the end of the sequence.

        Args:
            phase: ContactPhase to append.
        """
        self.phases.append(phase)

    def extend(self, other: "ContactSequence"):
        """Extend this sequence with phases from another sequence.

        Args:
            other: ContactSequence to append.
        """
        self.phases.extend(other.phases)

    def repeat(self, n: int) -> "ContactSequence":
        """Create a new sequence by repeating this one n times.

        Args:
            n: Number of repetitions.

        Returns:
            New ContactSequence with n copies of all phases.
        """
        new_phases = []
        for _ in range(n):
            new_phases.extend([p.copy() for p in self.phases])
        return ContactSequence(phases=new_phases)

    def scale_durations(self, factor: float) -> "ContactSequence":
        """Create a new sequence with all durations scaled.

        Args:
            factor: Scaling factor for all phase durations.

        Returns:
            New ContactSequence with scaled durations.
        """
        new_phases = []
        for phase in self.phases:
            new_phase = phase.copy()
            new_phase.duration *= factor
            new_phases.append(new_phase)
        return ContactSequence(phases=new_phases)

    def to_knot_schedule(self, dt: float) -> List[Tuple[int, ContactPhase]]:
        """Convert to a list of (num_knots, phase) for OCP construction.

        Discretizes each phase duration into integer number of knots.

        Args:
            dt: OCP timestep in seconds.

        Returns:
            List of (num_knots, phase) tuples.
        """
        schedule = []
        for phase in self.phases:
            num_knots = max(1, round(phase.duration / dt))
            schedule.append((num_knots, phase))
        return schedule

    def __iter__(self):
        """Iterate over phases."""
        return iter(self.phases)

    def __len__(self):
        """Return number of phases."""
        return len(self.phases)

    def __getitem__(self, index: int) -> ContactPhase:
        """Get phase by index."""
        return self.phases[index]
