#include <algorithm>
#include <cmath>
#include <cstdio>
#include "student_low_level_control.hpp"
int main() {
    StudentLowLevelControl controller;
    constexpr double dt = 0.01, target = 1.0;
    double speed = 0.0, peak = 0.0;
    for (int tick = 0; tick < 3000; ++tick) {
        const double rpm = llc::speedMpsToWheelRpm(speed, LLC_CFG.wheel_radius_m);
        const auto current = controller.step(0.0, target, rpm, rpm, 0.0, dt);
        const double torque = 0.5 * (current[0] / LLC_CFG.rear_left_iq_sign +
                                     current[1] / LLC_CFG.rear_right_iq_sign) * LLC_CFG.motor_torque_constant_nm_per_a;
        speed += (2.0 * torque / LLC_CFG.wheel_radius_m - 0.5 * speed) * dt / 2.0;
        peak = std::max({peak, std::abs(current[0]), std::abs(current[1])});
    }
    const auto turning = torque_vectoring_mnm(50.0, 1.0, 0.25, 0.0, LLC_CFG);
    controller.reset();
    const auto forward = controller.step(0.0, 1.0, 0.0, 0.0, 0.0, dt);
    const bool passed = std::abs(speed - target) < 0.08 &&
        peak <= LLC_CFG.torque_max_mnm * 0.001 / LLC_CFG.motor_torque_constant_nm_per_a + 1e-9 &&
        turning[0] < turning[1] && forward[0] > 0.0 && forward[1] < 0.0;
    std::printf("target %.2f m/s, final %.3f m/s; peak %.3f A\n", target, speed, peak);
    std::puts(passed ? "PASS - torque control, yaw vectoring and motor signs" : "FAIL - check torque/current steps");
    return passed ? 0 : 1;
}
