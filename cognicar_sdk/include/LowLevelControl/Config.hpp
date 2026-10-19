#pragma once

namespace cogni::low_level_control {

struct Config {
    double wheel_radius_m{0.033};
    double wheelbase_m{0.25};

    double kp{140.0};
    double ki{20.0};
    double kd{0.0};
    double integral_min{-40.0};
    double integral_max{40.0};

    double torque_min_mnm{-250.0};
    double torque_max_mnm{250.0};
    double torque_rate_limit_mnm_per_s{1000.0};

    double zero_speed_deadband_mps{0.2};
    double low_speed_feedforward_threshold_mps{0.3};
    double low_speed_feedforward_torque_mnm{40.0};

    bool torque_vectoring_enabled{true};
    double torque_vectoring_gain_mnm_per_radps{30.0};
    double torque_vectoring_max_delta_mnm{100.0};

    double motor_torque_constant_nm_per_a{0.01123};
    // The right motor is mounted mirrored: forward current is negative.
    double rear_left_iq_sign{1.0};
    double rear_right_iq_sign{-1.0};
};

}
