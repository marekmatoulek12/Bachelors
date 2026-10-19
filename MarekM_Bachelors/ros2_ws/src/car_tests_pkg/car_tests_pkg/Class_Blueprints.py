"""Twin-track vehicle model with Pacejka tires and traction ellipse limiting.

The model is intended for a small vehicle.  Each wheel can receive its own
drive torque, brake torque, and steering angle.  Wheel steering angles are
clamped to +/- 40 degrees.

State vector:
    x, y, yaw, vx, vy, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr

Units are SI: m, kg, s, rad, N, Nm.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TwinTrackParams:
    front_wheel_radius: float = 0.033
    rear_wheel_radius: float = 0.033
    front_axle_len: float = 0.16
    rear_axle_len: float = 0.14
    front_cornering_stiff: float = 16.0
    rear_cornering_stiff: float = 16.0
    mass: float = 2.5
    moment_of_inertia: float = 0.027
    st_min_velocity: float = 0.2

    # Extra parameters needed by a twin-track model.
    front_track_width: float = 0.16
    rear_track_width: float = 0.16
    wheel_inertia: float = 0.001
    tire_friction_coeff: float = 0.9
    rolling_resistance: float = 0.03
    # By was calibrated from recorded driving using rolling open-loop
    # predictions. Bx, C/E, and the friction coefficient retain their original
    # values because the recording did not identify them reliably.
    pacejka_bx: float = 10.0
    pacejka_cx: float = 1.9
    pacejka_ex: float = 0.97
    pacejka_by: float = 2.15
    pacejka_cy: float = 1.3
    pacejka_ey: float = -1.6
    max_steer_deg: float = 40.0
    gravity: float = 9.81


@dataclass
class TwinTrackState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    omega_fl: float = 0.0
    omega_fr: float = 0.0
    omega_rl: float = 0.0
    omega_rr: float = 0.0

@dataclass
class TwinTrackStateDerivatives:
    x_dot: float = 0.0
    y_dot: float = 0.0
    yaw_dot: float = 0.0
    vx_dot: float = 0.0
    vy_dot: float = 0.0
    yaw_rate_dot: float = 0.0
    omega_fl_dot: float = 0.0
    omega_fr_dot: float = 0.0
    omega_rl_dot: float = 0.0
    omega_rr_dot: float = 0.0

@dataclass(frozen=True)
class WheelForces:
    fx: float
    fy: float
    normal_force: float
    slip_ratio: float
    slip_angle: float

@dataclass
class WheelAllParams:
    fx: float
    fy: float
    normal_force: float
    slip_ratio: float
    slip_angle: float
    omega: float
    drive_torque: float
    break_torque: float


@dataclass(frozen=True)
class TwinTrackInput:
    """Per-wheel model inputs.

    Drive torque and brake torque are in Nm.  Steering angles are in radians.
    Brake torque is treated as a positive magnitude opposing wheel rotation.
    """

    drive_torque_fl: float = 0.0
    drive_torque_fr: float = 0.0
    drive_torque_rl: float = 0.0
    drive_torque_rr: float = 0.0
    brake_torque_fl: float = 0.0
    brake_torque_fr: float = 0.0
    brake_torque_rl: float = 0.0
    brake_torque_rr: float = 0.0
    steering_angle_fl: float = 0.0
    steering_angle_fr: float = 0.0
    steering_angle_rl: float = 0.0
    steering_angle_rr: float = 0.0


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0


@dataclass(frozen=True)
class RearTorqueCommand:
    left_torque: float
    right_torque: float
    slip_ratio_ref: float
    omega_ref_left: float
    omega_ref_right: float