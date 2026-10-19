// CogniCar estimation building blocks - the public student API for exercise 2.
//
// Same pattern as cogni_tracking: every function here has a compiled
// reference implementation in libstate_estimator_core.so. The student
// scaffold (student_kalman.{py,hpp}) calls these blocks and works out of the
// box; students replace ONE call at a time with their own implementation of
// the SAME signature.
//
// The filter: an extended Kalman filter over the car's pose.
//   state      [x, y, yaw]            map frame, m and rad
//   predict    wheel speed + gyro     every tick (dead reckoning)
//   update     pose fix               LiDAR on the car, simulator pose in --sim
// The fix measures the state directly (H = identity), which keeps the update
// equations short.
#ifndef STATE_ESTIMATOR_COGNIESTIMATION_HPP
#define STATE_ESTIMATOR_COGNIESTIMATION_HPP

#include <Eigen/Dense>

#include <CogniCommon/Export.hpp>

namespace cogni::state_estimator {

using Vector3 = Eigen::Vector3d;  // [x, y, yaw]
using Matrix3 = Eigen::Matrix3d;

struct PoseKfConfig {
    double wheel_radius_m{0.033};
    double dt_s{0.01};              // the autopilot's 100 Hz tick

    // Process noise added per predict: how far dead reckoning may drift per tick.
    double q_xy{2e-5};              // m^2
    double q_yaw{1e-5};             // rad^2

    // Measurement noise of the pose fix.
    double r_xy{2.5e-3};            // m^2   (5 cm standard deviation)
    double r_yaw{2.5e-3};           // rad^2 (~3 deg standard deviation)

    double initial_variance{1.0};
};

// Wraps an angle into [-pi, pi].
STATE_ESTIMATOR_API double wrapAngle(double angle_rad);

// Body speed from the two FRONT (undriven) wheel encoders: average, rpm -> m/s.
STATE_ESTIMATOR_API double speedFromWheels(double front_left_rpm,
                                            double front_right_rpm,
                                            double wheel_radius_m);

// Predict the state one tick ahead:
//   x' = x + v cos(yaw) dt ;  y' = y + v sin(yaw) dt ;  yaw' = yaw + w dt
STATE_ESTIMATOR_API Vector3 motionModel(const Vector3& state, double speed_mps,
                                         double yaw_rate_radps, double dt_s);

// Jacobian of motionModel with respect to the state:
//   F = [[1, 0, -v sin(yaw) dt], [0, 1, v cos(yaw) dt], [0, 0, 1]]
STATE_ESTIMATOR_API Matrix3 motionJacobian(const Vector3& state, double speed_mps,
                                            double dt_s);

// P' = F P F^T + Q
STATE_ESTIMATOR_API Matrix3 predictCovariance(const Matrix3& covariance,
                                               const Matrix3& jacobian,
                                               const Matrix3& process_noise);

// K = P (P + R)^-1
STATE_ESTIMATOR_API Matrix3 kalmanGain(const Matrix3& covariance,
                                        const Matrix3& measurement_noise);

// x' = x + K (z - x), with the yaw difference wrapped so that -179 deg vs
// +179 deg is a 2 deg correction, not 358 deg.
STATE_ESTIMATOR_API Vector3 updateState(const Vector3& state, const Matrix3& gain,
                                         const Vector3& measured);

// P' = (I - K) P
STATE_ESTIMATOR_API Matrix3 updateCovariance(const Matrix3& covariance,
                                              const Matrix3& gain);

}  // namespace cogni::state_estimator

#endif  // STATE_ESTIMATOR_COGNIESTIMATION_HPP
