"""Offline torque/current controller check; no hardware required."""
from student_low_level_control import CFG, StudentLowLevelControl, torque_vectoring_mnm
import cogni_low_level_control as llc


def main():
    controller = StudentLowLevelControl()
    dt, target, speed, peak = 0.01, 1.0, 0.0, 0.0
    for _ in range(3000):
        rpm = llc.speed_mps_to_wheel_rpm(speed, CFG.wheel_radius_m)
        left, right = controller.step(0.0, target, rpm, rpm, 0.0, dt)
        torque = 0.5 * (left / CFG.rear_left_iq_sign + right / CFG.rear_right_iq_sign) * CFG.motor_torque_constant_nm_per_a
        speed += (2.0 * torque / CFG.wheel_radius_m - 0.5 * speed) * dt / 2.0
        peak = max(peak, abs(left), abs(right))
    turning = torque_vectoring_mnm(50.0, 1.0, 0.25, 0.0, CFG)
    controller.reset()
    forward = controller.step(0.0, 1.0, 0.0, 0.0, 0.0, dt)
    passed = (abs(speed - target) < 0.08 and
              peak <= CFG.torque_max_mnm * 0.001 / CFG.motor_torque_constant_nm_per_a + 1e-9 and
              turning[0] < turning[1] and forward[0] > 0 and forward[1] < 0)
    print(f"target {target:.2f} m/s, final {speed:.3f} m/s; peak {peak:.3f} A")
    print("PASS - torque control, yaw vectoring and motor signs" if passed else "FAIL - check torque/current steps")
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
