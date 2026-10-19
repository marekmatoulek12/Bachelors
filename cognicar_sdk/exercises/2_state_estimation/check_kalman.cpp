// Exercise 2 check: run YOUR filter offline against a known truth.
//
// No car, no simulation. The car drives a 1.5 m circle at 1 m/s. The wheel
// speed is 3 % off and noisy, the gyro has a bias, and the pose fix (10 Hz,
// 5 cm / 3 deg noise) drops out between 15 s and 18 s - watch the uncertainty
// grow while the filter dead-reckons. Python twin: check_kalman.py.
//
// Build:  cmake --build build --target check_kalman
// Run:    ./build/check_kalman
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "student_kalman.hpp"   // the filter you edit

int main()
{
    constexpr double kRadiusM = 1.5;
    constexpr double kSpeedMps = 1.0;
    constexpr int kTicks = 3000;      // 30 s at 100 Hz
    constexpr int kFixEvery = 10;     // 10 Hz pose fix
    const double dt = KF_CFG.dt_s;
    const double yaw_rate = kSpeedMps / kRadiusM;
    const double rpm = kSpeedMps / KF_CFG.wheel_radius_m * 60.0 / (2.0 * M_PI);

    std::mt19937 rng(7);
    std::normal_distribution<double> wheel_noise(0.0, 15.0), gyro_noise(0.0, 0.03),
        fix_xy_noise(0.0, 0.05), fix_yaw_noise(0.0, 0.05);

    StudentKalman kf;
    double err_raw_xy = 0.0, err_kf_xy = 0.0, err_raw_yaw = 0.0, err_kf_yaw = 0.0, worst = 0.0;
    int fixes = 0;

    for (int tick = 0; tick <= kTicks; ++tick) {
        const double t = tick * dt;
        const double true_yaw = se::wrapAngle(yaw_rate * t);
        const double true_x = kRadiusM * std::sin(yaw_rate * t);
        const double true_y = kRadiusM * (1.0 - std::cos(yaw_rate * t));

        if (tick > 0) {
            kf.predict(1.03 * rpm + wheel_noise(rng), 1.03 * rpm + wheel_noise(rng),
                       yaw_rate + 0.02 + gyro_noise(rng));
        }

        const bool dropout = t >= 15.0 && t < 18.0;
        if (tick % kFixEvery == 0 && !dropout) {
            const double zx = true_x + fix_xy_noise(rng);
            const double zy = true_y + fix_xy_noise(rng);
            const double zyaw = se::wrapAngle(true_yaw + fix_yaw_noise(rng));
            if (tick == 0) {
                kf.reset(zx, zy, zyaw);
            } else {
                kf.update(zx, zy, zyaw);
                err_raw_xy += std::pow(std::hypot(zx - true_x, zy - true_y), 2);
                err_kf_xy += std::pow(std::hypot(kf.x() - true_x, kf.y() - true_y), 2);
                err_raw_yaw += std::pow(se::wrapAngle(zyaw - true_yaw), 2);
                err_kf_yaw += std::pow(se::wrapAngle(kf.yaw() - true_yaw), 2);
                ++fixes;
            }
        }

        const double err = std::hypot(kf.x() - true_x, kf.y() - true_y);
        worst = std::max(worst, err);
        if (tick % 300 == 0) {
            std::printf("t=%4.1fs  position error %5.3f m  heading error %5.1f deg  "
                        "sigma_x %5.3f m%s\n",
                        t, err, se::wrapAngle(kf.yaw() - true_yaw) * 180.0 / M_PI,
                        std::sqrt(kf.covariance()(0, 0)), dropout ? "  (no fix)" : "");
        }
    }

    const double rms_raw_xy = std::sqrt(err_raw_xy / fixes);
    const double rms_kf_xy = std::sqrt(err_kf_xy / fixes);
    const double rms_raw_yaw = std::sqrt(err_raw_yaw / fixes) * 180.0 / M_PI;
    const double rms_kf_yaw = std::sqrt(err_kf_yaw / fixes) * 180.0 / M_PI;
    std::printf("\nRMS position   raw fix %.3f m    your filter %.3f m\n", rms_raw_xy, rms_kf_xy);
    std::printf("RMS heading    raw fix %.1f deg   your filter %.1f deg\n", rms_raw_yaw, rms_kf_yaw);
    std::printf("worst position error (incl. the 3 s dropout)  %.3f m\n", worst);

    const bool pass = rms_kf_xy < rms_raw_xy && rms_kf_yaw < rms_raw_yaw && worst < 0.3;
    std::puts(pass ? "PASS - your filter beats the raw fix and survives the dropout"
                   : "FAIL - check the step you changed last against its formula");
    return pass ? 0 : 1;
}
