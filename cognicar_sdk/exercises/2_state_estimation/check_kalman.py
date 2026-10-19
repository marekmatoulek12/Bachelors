"""Exercise 2 check: run YOUR filter offline against a known truth.

No car, no simulation. The car drives a 1.5 m circle at 1 m/s. The wheel speed
is 3 % off and noisy, the gyro has a bias, and the pose fix (10 Hz, 5 cm / 3 deg
noise) drops out between 15 s and 18 s - watch the uncertainty grow while the
filter dead-reckons. C++ twin: check_kalman.cpp.

    python3 check_kalman.py
"""
import math
import os
import random
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from student_kalman import CFG, StudentKalman, se  # noqa: E402  (this is YOUR filter)

RADIUS_M = 1.5
SPEED_MPS = 1.0
TICKS = 3000       # 30 s at 100 Hz
FIX_EVERY = 10     # 10 Hz pose fix


def main():
    random.seed(7)
    dt = CFG.dt_s
    yaw_rate = SPEED_MPS / RADIUS_M
    rpm = SPEED_MPS / CFG.wheel_radius_m * 60.0 / (2.0 * math.pi)

    kf = StudentKalman()
    err_raw_xy = err_kf_xy = err_raw_yaw = err_kf_yaw = worst = 0.0
    fixes = 0

    for tick in range(TICKS + 1):
        t = tick * dt
        true_yaw = se.wrap_angle(yaw_rate * t)
        true_x = RADIUS_M * math.sin(yaw_rate * t)
        true_y = RADIUS_M * (1.0 - math.cos(yaw_rate * t))

        if tick > 0:
            kf.predict(1.03 * rpm + random.gauss(0, 15), 1.03 * rpm + random.gauss(0, 15),
                       yaw_rate + 0.02 + random.gauss(0, 0.03))

        dropout = 15.0 <= t < 18.0
        if tick % FIX_EVERY == 0 and not dropout:
            zx = true_x + random.gauss(0, 0.05)
            zy = true_y + random.gauss(0, 0.05)
            zyaw = se.wrap_angle(true_yaw + random.gauss(0, 0.05))
            if tick == 0:
                kf.reset(zx, zy, zyaw)
            else:
                kf.update(zx, zy, zyaw)
                err_raw_xy += math.hypot(zx - true_x, zy - true_y) ** 2
                err_kf_xy += math.hypot(kf.x - true_x, kf.y - true_y) ** 2
                err_raw_yaw += se.wrap_angle(zyaw - true_yaw) ** 2
                err_kf_yaw += se.wrap_angle(kf.yaw - true_yaw) ** 2
                fixes += 1

        err = math.hypot(kf.x - true_x, kf.y - true_y)
        worst = max(worst, err)
        if tick % 300 == 0:
            print(f"t={t:4.1f}s  position error {err:5.3f} m  "
                  f"heading error {math.degrees(se.wrap_angle(kf.yaw - true_yaw)):5.1f} deg  "
                  f"sigma_x {math.sqrt(kf.P[0, 0]):5.3f} m" + ("  (no fix)" if dropout else ""))

    rms_raw_xy = math.sqrt(err_raw_xy / fixes)
    rms_kf_xy = math.sqrt(err_kf_xy / fixes)
    rms_raw_yaw = math.degrees(math.sqrt(err_raw_yaw / fixes))
    rms_kf_yaw = math.degrees(math.sqrt(err_kf_yaw / fixes))
    print(f"\nRMS position   raw fix {rms_raw_xy:.3f} m    your filter {rms_kf_xy:.3f} m")
    print(f"RMS heading    raw fix {rms_raw_yaw:.1f} deg   your filter {rms_kf_yaw:.1f} deg")
    print(f"worst position error (incl. the 3 s dropout)  {worst:.3f} m")

    ok = rms_kf_xy < rms_raw_xy and rms_kf_yaw < rms_raw_yaw and worst < 0.3
    print("PASS - your filter beats the raw fix and survives the dropout" if ok
          else "FAIL - check the step you changed last against its formula")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
