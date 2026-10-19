#pragma once
#include <algorithm>
#include <cmath>
#include <LowLevelControl/Steps.hpp>

namespace llc = cogni::low_level_control;
using LowLevelControlConfig = llc::Config;
inline LowLevelControlConfig LLC_CFG{};

// Replace each wrapper with your implementation; keep the physical units.
inline double wheel_rpm_to_speed_mps(double rpm, double radius) {
    return llc::wheelRpmToSpeedMps(rpm, radius);
}
inline double pid_torque_mnm(double error, double dt, double& previous, double& integral,
                             const LowLevelControlConfig& cfg, bool integrate = true) {
    return llc::pidTorqueMnm(error, dt, previous, integral, cfg, integrate);
}
inline double rate_limit_torque_mnm(double requested, double previous, double dt, const LowLevelControlConfig& cfg) {
    return llc::rateLimitTorqueMnm(requested, previous, dt, cfg);
}
inline llc::RearTorqueMnm torque_vectoring_mnm(double base, double speed, double steer, double yaw,
                                             const LowLevelControlConfig& cfg) {
    return llc::torqueVectoringMnm(base, speed, steer, yaw, cfg);
}
inline double torque_mnm_to_current_a(double torque, const LowLevelControlConfig& cfg) {
    return llc::torqueMnmToCurrentA(torque, cfg);
}

class StudentLowLevelControl {
public:
    explicit StudentLowLevelControl(const LowLevelControlConfig& config = LLC_CFG) : config_(config) {}
    void reset() {
        previous_error_mps_ = 0.0;
        integral_error_mps_s_ = 0.0;
        previous_base_torque_mnm_ = 0.0;
    }
    llc::RearCurrentA step(double steer_rad, double target_speed_mps, double front_left_rpm,
                          double front_right_rpm, double yaw_rate_radps, double dt_s) {
        if (!llc::validConfig(config_) || !std::isfinite(target_speed_mps) ||
            !std::isfinite(steer_rad) || !std::isfinite(front_left_rpm) ||
            !std::isfinite(front_right_rpm) || !std::isfinite(yaw_rate_radps) ||
            !std::isfinite(dt_s) || dt_s <= 0.0 ||
            !std::isfinite(config_.torque_rate_limit_mnm_per_s * dt_s)) {
            reset();
            return {0.0, 0.0};
        }
        const double speed_mps = wheel_rpm_to_speed_mps(
            0.5 * front_left_rpm + 0.5 * front_right_rpm, config_.wheel_radius_m);
        const bool stop = std::abs(target_speed_mps) <= config_.zero_speed_deadband_mps;
        const double target = stop ? 0.0 : target_speed_mps;
        const double error = target - speed_mps;
        if (!std::isfinite(speed_mps) || !std::isfinite(error) ||
            !std::isfinite(speed_mps * std::tan(steer_rad) / config_.wheelbase_m)) {
            reset();
            return {0.0, 0.0};
        }
        double base = pid_torque_mnm(error, dt_s, previous_error_mps_, integral_error_mps_s_, config_, !stop);
        if (!stop && std::abs(speed_mps) < config_.low_speed_feedforward_threshold_mps) {
            base += std::copysign(config_.low_speed_feedforward_torque_mnm, target);
        }
        base = rate_limit_torque_mnm(std::clamp(base, config_.torque_min_mnm, config_.torque_max_mnm),
                                 previous_base_torque_mnm_, dt_s, config_);
        previous_base_torque_mnm_ = base;
        const auto torque = torque_vectoring_mnm(base, speed_mps, steer_rad, yaw_rate_radps, config_);
        const llc::RearCurrentA current{config_.rear_left_iq_sign * torque_mnm_to_current_a(torque[0], config_),
                                   config_.rear_right_iq_sign * torque_mnm_to_current_a(torque[1], config_)};
        if (!std::isfinite(current[0]) || !std::isfinite(current[1])) {
            reset();
            return {0.0, 0.0};
        }
        return current;
    }
private:
    LowLevelControlConfig config_{};
    double previous_error_mps_{0.0}, integral_error_mps_s_{0.0}, previous_base_torque_mnm_{0.0};
};
