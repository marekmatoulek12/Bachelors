"""Donkey-and-carrot pure pursuit - student scaffold (Python).

It works as-is: every step below just calls the reference implementation
(`tt.*`). Your job is to make the tracker your own, ONE STEP AT A TIME:

    1. pick a function in the "IMPLEMENT THESE" block below
    2. replace its body with your own math, keeping the SAME signature
    3. re-run and check it still drives well against the DigitalTwin
    4. repeat until every step is yours -- then you've written the tracker

Never edit the "FIXED PIPELINE" section: it always calls YOUR functions, so the
tracker keeps running no matter how many steps you've rewritten so far.

Run it:   python3 student_tracker.py     (from the SDK root, so data/ resolves)
"""
import math

import cogni_tracking as tt

CFG = tt.Config()  # gains & limits - tweak freely

# Tuned for the CogniCar oval track (matches config/pure_pursuit.yaml, the
# same tuning path_tracking_node runs with). The class defaults above are
# untuned and oscillate more; start from these instead.
CFG.lookahead_distance = 0.45        # was 0.3
CFG.lookahead_velocity_gain = 0.35   # was 0.15
CFG.kp_heading = 1.6                 # was 2.5
CFG.kp_cross_track = 0.8             # was 1.0
CFG.dt_s = 0.01                      # was 0.05; must match the autopilot's 100 Hz loop


# ============================================================================
#  IMPLEMENT THESE, ONE AT A TIME.  Each starts by calling the reference tt.*;
#  swap in your own body but keep the signature identical.
# ============================================================================

def project_to_path(path, x, y):
    """Nearest point on the path -> projection (has .cross_track_error_m)."""
    return tt.project_to_path(path, x, y)                       # TODO: your own


def lookahead_point(path, x, y, lookahead_m):
    """The 'carrot' a bit ahead of the car on the path -> (cx, cy)."""
    return tt.lookahead_point(path, x, y, lookahead_m)          # TODO: your own


def heading_error(carrot_x, carrot_y, x, y, yaw_rad):
    """Angle from the car's heading to the carrot, and the carrot distance.

    Returns (error_rad, carrot_distance_m).
    """
    return tt.heading_error(carrot_x, carrot_y, x, y, yaw_rad)  # TODO: your own


def steering_from_errors(heading_error_rad, carrot_distance_m, cross_track_error_m, cfg):
    """Raw steering angle (rad) from the errors, before clamp/slew."""
    return tt.steering_from_errors(heading_error_rad, carrot_distance_m,
                                   cross_track_error_m, cfg)     # TODO: your own


def clamp_steering(steering_rad, cfg):
    """Keep the steering within the servo's physical limits."""
    lo = cfg.min_steering_rad
    hi = cfg.max_steering_rad
    return max(lo, min(hi, steering_rad))                       # TODO: your own


def rate_limit(steering_rad, previous_steering_rad, max_rate_radps, dt_s):
    """Stop the steering jumping faster than the servo can move (slew limit)."""
    return tt.rate_limit(steering_rad, previous_steering_rad,
                         max_rate_radps, dt_s)                   # TODO: your own


# ============================================================================
#  FIXED PIPELINE - do not edit. It wires your functions together and keeps the
#  one bit of state a real controller needs (the previous steering command, for
#  the slew limit). Call reset() when a run restarts.
# ============================================================================

class StudentTracker:
    def __init__(self):
        self._prev_steer_rad = 0.0

    def reset(self):
        self._prev_steer_rad = 0.0

    def step(self, path, pose, twist):
        """Returns the steering command in RADIANS.

        Takes the car;'s full ROS-style pose (position + orientation) and
        twist (linear + angular velocity). Today's pipeline reads x, y, yaw
        and the forward speed - but everything else (e.g. twist.angular.z,
        the yaw rate) arrives here too, ready when your controller wants it.
        """
        x = pose.position.x
        y = pose.position.y
        o = pose.orientation
        yaw = tt.yaw_rad(o.x, o.y, o.z, o.w)
        speed_mps = twist.linear.x

        proj = project_to_path(path, x, y)

        lookahead = min(CFG.lookahead_distance + CFG.lookahead_velocity_gain * speed_mps,
                        CFG.max_lookahead_distance)
        cx, cy = lookahead_point(path, x, y, lookahead)

        he, carrot_dist = heading_error(cx, cy, x, y, yaw)
        steer = steering_from_errors(he, carrot_dist, proj.cross_track_error_m, CFG)

        steer = clamp_steering(steer, CFG)
        steer = rate_limit(steer, self._prev_steer_rad,
                           CFG.max_steering_rate_radps, CFG.dt_s)
        self._prev_steer_rad = steer

        return steer
