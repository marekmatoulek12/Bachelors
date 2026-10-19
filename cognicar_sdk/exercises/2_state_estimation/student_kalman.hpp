// Pose Kalman filter - student filter (C++).
//
// The autopilot drives on THIS filter's pose, the same way it steers with your
// tracker. It works as-is: every step below calls the reference se:: block.
// Make the filter your own, ONE STEP AT A TIME:
//
//     1. pick a function in the "IMPLEMENT THESE" block below
//     2. replace its body with your own math, keeping the SAME signature
//     3. run check_kalman (offline, instant), then rebuild autopilot and
//        check it still drives
//     4. repeat until every step is yours -- then you've written the filter
//
// Never edit the "FIXED PIPELINE" section: it always calls YOUR functions.
//
// The filter in one paragraph: the state is the pose [x, y, yaw] plus a 3x3
// covariance P (how unsure we are). Every tick, PREDICT moves the pose with the
// wheel speed and the gyro and lets P grow. Whenever a pose fix arrives
// (LiDAR on the car, the simulator in --sim), UPDATE pulls the pose toward it,
// weighted by the Kalman gain K, and P shrinks.
//
// Vectors and matrices are Eigen: A * B, A.transpose(), A.inverse(),
// Matrix3::Identity(), v[0]. Python twin: student_kalman.py.
#pragma once

#include <StateEstimator/CogniEstimation.hpp>

namespace se = cogni::state_estimator;
using se::Matrix3;
using se::Vector3;

inline se::PoseKfConfig KF_CFG;  // noise levels - tweak freely


// ============================================================================
//  IMPLEMENT THESE, ONE AT A TIME.  Each starts by calling the reference se::;
//  swap in your own body but keep the signature identical.
// ============================================================================

// Forward speed (m/s) from the two front wheels (rpm). They are not driven, so
// they measure ground speed, not wheel spin.
inline double speed_from_wheels(double front_left_rpm, double front_right_rpm,
                                double wheel_radius_m)
{
    return se::speedFromWheels(front_left_rpm, front_right_rpm, wheel_radius_m);  // TODO: your own
}

// Move the pose one tick: drive `speed` along the heading, turn by the gyro.
inline Vector3 motion_model(const Vector3& state, double speed_mps,
                            double yaw_rate_radps, double dt_s)
{
    return se::motionModel(state, speed_mps, yaw_rate_radps, dt_s);  // TODO: your own
}

// How the new pose changes when the old one changes: d motion_model / d state.
inline Matrix3 motion_jacobian(const Vector3& state, double speed_mps, double dt_s)
{
    return se::motionJacobian(state, speed_mps, dt_s);  // TODO: your own
}

// Grow the uncertainty: P' = F P F^T + Q.
inline Matrix3 predict_covariance(const Matrix3& P, const Matrix3& F, const Matrix3& Q)
{
    return se::predictCovariance(P, F, Q);  // TODO: your own
}

// How much to trust the fix: K = P (P + R)^-1.
inline Matrix3 kalman_gain(const Matrix3& P, const Matrix3& R)
{
    return se::kalmanGain(P, R);  // TODO: your own
}

// Pull the pose toward the fix: x' = x + K (z - x). Wrap the yaw difference
// (se::wrapAngle) - otherwise -179 deg vs +179 deg looks like a full turn.
inline Vector3 update_state(const Vector3& state, const Matrix3& K, const Vector3& z)
{
    return se::updateState(state, K, z);  // TODO: your own
}

// Shrink the uncertainty: P' = (I - K) P.
inline Matrix3 update_covariance(const Matrix3& P, const Matrix3& K)
{
    return se::updateCovariance(P, K);  // TODO: your own
}


// ============================================================================
//  FIXED PIPELINE - do not edit. It wires your functions together.
// ============================================================================

class StudentKalman {
public:
    // Start at a known pose, e.g. the first fix.
    void reset(double x, double y, double yaw_rad)
    {
        state_ = Vector3(x, y, yaw_rad);
        P_ = KF_CFG.initial_variance * Matrix3::Identity();
    }

    // Every tick.
    void predict(double front_left_rpm, double front_right_rpm, double yaw_rate_radps)
    {
        speed_mps_ = speed_from_wheels(front_left_rpm, front_right_rpm, KF_CFG.wheel_radius_m);
        yaw_rate_radps_ = yaw_rate_radps;
        const Matrix3 F = motion_jacobian(state_, speed_mps_, KF_CFG.dt_s);
        state_ = motion_model(state_, speed_mps_, yaw_rate_radps_, KF_CFG.dt_s);
        const Matrix3 Q = Vector3(KF_CFG.q_xy, KF_CFG.q_xy, KF_CFG.q_yaw).asDiagonal();
        P_ = predict_covariance(P_, F, Q);
    }

    // Whenever a NEW pose fix arrives.
    void update(double x, double y, double yaw_rad)
    {
        const Matrix3 R = Vector3(KF_CFG.r_xy, KF_CFG.r_xy, KF_CFG.r_yaw).asDiagonal();
        const Matrix3 K = kalman_gain(P_, R);
        state_ = update_state(state_, K, Vector3(x, y, yaw_rad));
        P_ = update_covariance(P_, K);
    }

    double x() const { return state_[0]; }
    double y() const { return state_[1]; }
    double yaw() const { return state_[2]; }
    double speed() const { return speed_mps_; }
    double yaw_rate() const { return yaw_rate_radps_; }
    const Matrix3& covariance() const { return P_; }

private:
    Vector3 state_{Vector3::Zero()};
    Matrix3 P_{Matrix3::Identity()};
    double speed_mps_{0.0};
    double yaw_rate_radps_{0.0};
};
