"""Torque/current control scaffold. Replace individual wrappers with your code."""
import math
import cogni_low_level_control as llc

CFG = llc.Config()


def wheel_rpm_to_speed_mps(rpm, radius):
    return llc.wheel_rpm_to_speed_mps(rpm, radius)


def pid_torque_mnm(error, dt, previous, integral, cfg, integrate=True):
    return llc.pid_torque_mnm(error, dt, previous, integral, cfg, integrate)


def rate_limit_torque_mnm(requested, previous, dt, cfg):
    return llc.rate_limit_torque_mnm(requested, previous, dt, cfg)


def torque_vectoring_mnm(base, speed, steer, yaw, cfg):
    return llc.torque_vectoring_mnm(base, speed, steer, yaw, cfg)


def torque_mnm_to_current_a(torque, cfg):
    return llc.torque_mnm_to_current_a(torque, cfg)


class LowLevelControl:
    """Returns rear motor currents [A], including the mounting signs."""
    def __init__(self, config=None):
        self.config = config if config is not None else CFG
        self.reset()

    def reset(self):
        self.previous_error = 0.0
        self.integral_error = 0.0
        self.previous_base = 0.0

    def step(self, steer_rad, target_speed_mps, front_left_rpm,
             front_right_rpm, yaw_rate_radps, dt_s):
        cfg = self.config
        if (not llc.valid_config(cfg) or dt_s <= 0.0 or
                not all(math.isfinite(v) for v in
                        (steer_rad, target_speed_mps, front_left_rpm, front_right_rpm,
                         yaw_rate_radps, dt_s, cfg.torque_rate_limit_mnm_per_s * dt_s))):
            self.reset()
            return (0.0, 0.0)
        speed = wheel_rpm_to_speed_mps(0.5 * front_left_rpm + 0.5 * front_right_rpm, cfg.wheel_radius_m)
        stop = abs(target_speed_mps) <= cfg.zero_speed_deadband_mps
        target = 0.0 if stop else target_speed_mps
        error = target - speed
        if not all(math.isfinite(v) for v in (speed, error, speed * math.tan(steer_rad) / cfg.wheelbase_m)):
            self.reset()
            return (0.0, 0.0)
        base, self.previous_error, self.integral_error = pid_torque_mnm(
            error, dt_s, self.previous_error, self.integral_error, cfg, not stop)
        if not stop and abs(speed) < cfg.low_speed_feedforward_threshold_mps:
            base += math.copysign(cfg.low_speed_feedforward_torque_mnm, target)
        base = rate_limit_torque_mnm(min(cfg.torque_max_mnm, max(cfg.torque_min_mnm, base)),
                                     self.previous_base, dt_s, cfg)
        self.previous_base = base
        left, right = torque_vectoring_mnm(base, speed, steer_rad, yaw_rate_radps, cfg)
        current = (cfg.rear_left_iq_sign * torque_mnm_to_current_a(left, cfg),
                   cfg.rear_right_iq_sign * torque_mnm_to_current_a(right, cfg))
        if not all(math.isfinite(v) for v in current):
            self.reset()
            return (0.0, 0.0)
        return current
