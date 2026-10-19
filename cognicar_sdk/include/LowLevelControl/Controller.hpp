#pragma once

#include <LowLevelControl/Steps.hpp>

namespace cogni::low_level_control {

class LOW_LEVEL_CONTROL_API Controller {
public:
    Controller();
    explicit Controller(const Config& config);
    void setConfig(const Config& config);
    const Config& config() const;
    void reset();
    RearCurrentA step(double steer_rad, double target_speed_mps, double front_left_rpm,
                      double front_right_rpm, double yaw_rate_radps, double dt_s);
private:
    Config config_{};
    double previous_error_mps_{0.0};
    double integral_error_mps_s_{0.0};
    double previous_base_torque_mnm_{0.0};
};

}
