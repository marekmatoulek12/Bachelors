"""Pose Kalman filter - student filter (Python).

The autopilot drives on THIS filter's pose, the same way it steers with your
tracker. It works as-is: every step below calls the reference `se.*` block.
Make the filter your own, ONE STEP AT A TIME:

    1. pick a function in the "IMPLEMENT THESE" block below
    2. replace its body with your own math, keeping the SAME signature
    3. run check_kalman.py (offline, instant), then check the autopilot still drives
    4. repeat until every step is yours -- then you've written the filter

Never edit the "FIXED PIPELINE" section: it always calls YOUR functions.

The filter in one paragraph: the state is the pose [x, y, yaw] plus a 3x3
covariance P (how unsure we are). Every tick, PREDICT moves the pose with the
wheel speed and the gyro and lets P grow. Whenever a pose fix arrives (LiDAR on
the car, the simulator in --sim), UPDATE pulls the pose toward it, weighted by
the Kalman gain K, and P shrinks.

Vectors and matrices are numpy arrays: A @ B, A.T, np.linalg.inv(A), np.eye(3).
C++ twin: student_kalman.hpp.
"""
import numpy as np

import cogni_state_estimator as se

CFG = se.PoseKfConfig()  # noise levels - tweak freely


# ============================================================================
#  IMPLEMENT THESE, ONE AT A TIME.  Each starts by calling the reference se.*;
#  swap in your own body but keep the signature identical.
# ============================================================================

def speed_from_wheels(front_left_rpm, front_right_rpm, wheel_radius_m):
    """Forward speed (m/s) from the two front wheels (rpm).

    They are not driven, so they measure ground speed, not wheel spin.
    """
    return se.speed_from_wheels(front_left_rpm, front_right_rpm, wheel_radius_m)  # TODO: your own


def motion_model(state, speed_mps, yaw_rate_radps, dt_s):
    """Move the pose [x, y, yaw] one tick: drive along the heading, turn by the gyro."""
    return se.motion_model(state, speed_mps, yaw_rate_radps, dt_s)  # TODO: your own


def motion_jacobian(state, speed_mps, dt_s):
    """How the new pose changes when the old one changes: d motion_model / d state (3x3)."""
    return se.motion_jacobian(state, speed_mps, dt_s)  # TODO: your own


def predict_covariance(P, F, Q):
    """Grow the uncertainty: P' = F P F^T + Q."""
    return se.predict_covariance(P, F, Q)  # TODO: your own


def kalman_gain(P, R):
    """How much to trust the fix: K = P (P + R)^-1."""
    return se.kalman_gain(P, R)  # TODO: your own


def update_state(state, K, z):
    """Pull the pose toward the fix: x' = x + K (z - x).

    Wrap the yaw difference (se.wrap_angle) - otherwise -179 deg vs +179 deg
    looks like a full turn.
    """
    return se.update_state(state, K, z)  # TODO: your own


def update_covariance(P, K):
    """Shrink the uncertainty: P' = (I - K) P."""
    return se.update_covariance(P, K)  # TODO: your own


# ============================================================================
#  FIXED PIPELINE - do not edit. It wires your functions together.
# ============================================================================

class StudentKalman:
    def __init__(self):
        self.state = np.zeros(3)
        self.P = np.eye(3)
        self.speed = 0.0
        self.yaw_rate = 0.0

    def reset(self, x, y, yaw_rad):
        """Start at a known pose, e.g. the first fix."""
        self.state = np.array([x, y, yaw_rad])
        self.P = CFG.initial_variance * np.eye(3)

    def predict(self, front_left_rpm, front_right_rpm, yaw_rate_radps):
        """Every tick."""
        self.speed = speed_from_wheels(front_left_rpm, front_right_rpm, CFG.wheel_radius_m)
        self.yaw_rate = yaw_rate_radps
        F = np.asarray(motion_jacobian(self.state, self.speed, CFG.dt_s))
        self.state = np.asarray(motion_model(self.state, self.speed, self.yaw_rate, CFG.dt_s))
        Q = np.diag([CFG.q_xy, CFG.q_xy, CFG.q_yaw])
        self.P = np.asarray(predict_covariance(self.P, F, Q))

    def update(self, x, y, yaw_rad):
        """Whenever a NEW pose fix arrives."""
        R = np.diag([CFG.r_xy, CFG.r_xy, CFG.r_yaw])
        K = np.asarray(kalman_gain(self.P, R))
        self.state = np.asarray(update_state(self.state, K, np.array([x, y, yaw_rad])))
        self.P = np.asarray(update_covariance(self.P, K))

    @property
    def x(self):
        return self.state[0]

    @property
    def y(self):
        return self.state[1]

    @property
    def yaw(self):
        return self.state[2]
