#pragma once

#include <array>
#include <LowLevelControl/Config.hpp>
#include <CogniCommon/Export.hpp>

namespace cogni::low_level_control {

using RearCurrentA = std::array<double, 2>;
using RearTorqueMnm = std::array<double, 2>;

LOW_LEVEL_CONTROL_API bool validConfig(const Config& config);
LOW_LEVEL_CONTROL_API double wheelRpmToSpeedMps(double wheel_rpm, double wheel_radius_m);
LOW_LEVEL_CONTROL_API double speedMpsToWheelRpm(double speed_mps, double wheel_radius_m);
LOW_LEVEL_CONTROL_API double speedErrorMps(double target_mps, double left_rpm, double right_rpm, const Config& config);
LOW_LEVEL_CONTROL_API double pidTorqueMnm(double error_mps, double dt_s, double& previous_error_mps,
                                         double& integral_error_mps_s, const Config& config, bool integrate = true);
LOW_LEVEL_CONTROL_API double rateLimitTorqueMnm(double requested, double previous, double dt_s, const Config& config);
LOW_LEVEL_CONTROL_API RearTorqueMnm torqueVectoringMnm(double base_torque_mnm, double speed_mps,
                                                       double steer_rad, double yaw_rate_radps, const Config& config);
LOW_LEVEL_CONTROL_API double torqueMnmToCurrentA(double torque_mnm, const Config& config);

}
