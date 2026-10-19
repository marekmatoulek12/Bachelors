// CogniCar state estimator - the public student API.
//
// The same estimator architecture that runs on the real car - an extended
// Kalman filter over a vehicle model - with the SIMPLE kinematic bicycle
// model (handout 1) as the process model. It fuses
//
//   predict      wheel-odometry speed + steering request  (the model)
//   updatePose   an external pose fix, e.g. /map/pose     (slow, absolute)
//   updateYawRate the IMU gyro                            (fast, relative)
//
// into one smooth vehicle state: position, yaw, yaw rate and side-slip angle.
// The implementation ships compiled only; this header is the whole contract.
#ifndef STATE_ESTIMATOR_STATEESTIMATOR_HPP
#define STATE_ESTIMATOR_STATEESTIMATOR_HPP

#include <memory>

#include <CogniCommon/Export.hpp>

namespace cogni::state_estimator {

// The vehicle model the filter predicts with. Kinematic is the simple
// handout-1 bicycle; SingleTrack is the dynamic (tire) model the real car
// runs. Both use the same state (beta, yawrate, x, y, yaw), so they are
// freely switchable.
enum class VehicleModel {
    Kinematic,
    SingleTrack,
};

// Everything the estimator assumes about the car and the sensors.
// Defaults match the real CogniCar configuration.
struct EstimatorConfig {
    VehicleModel model{VehicleModel::Kinematic};

    // --- vehicle geometry ----------------------------------------------------
    double front_wheel_radius_m{0.033};    // for the wheel-odometry rpm -> m/s
    double rear_wheel_radius_m{0.033};     // single-track model only; equal to
                                           // the front radius on the CogniCar
    double front_axle_len_m{0.155};        // CG -> front axle
    double rear_axle_len_m{0.100};         // CG -> rear axle

    // steering request -> effective front wheel angle
    double steering_gain{0.5};
    double steering_bias_rad{0.0};
    double min_steering_rad{-0.7};
    double max_steering_rad{0.7};
    double steering_smoothing{0.75};       // exponential smoothing of the request (0 = off)

    // --- Kinematic model only -------------------------------------------------
    // How quickly side slip and yaw rate settle onto the kinematic geometry
    // (they are filter states so the gyro can correct them).
    double model_time_constant_s{0.1};

    // --- SingleTrack model only -----------------------------------------------
    double front_cornering_stiffness{46.0};
    double rear_cornering_stiffness{52.0};
    double vehicle_mass_kg{2.5};
    double yaw_inertia_kgm2{0.03};
    double min_velocity_mps{0.3};          // below this the tire dynamics freeze

    // --- filter tuning --------------------------------------------------------
    // Process noise: how much the model may be wrong per prediction step.
    double q_side_slip{0.15};
    double q_yaw_rate{0.5};
    double q_x{0.1};
    double q_y{0.1};
    double q_yaw{0.1};

    // Measurement noise: how much each sensor may be wrong.
    double r_pose_x{5.0};
    double r_pose_y{5.0};
    double r_pose_yaw{2.0};
    double r_yaw_rate{1.2};

    double dt_s{0.02};                     // prediction period (50 Hz, like the car)
};

// The fused vehicle state in the map frame.
struct EstimatedState {
    double x{};                 // m
    double y{};                 // m
    double yaw_rad{};           // heading
    double yaw_rate_radps{};    // turn rate
    double side_slip_rad{};     // beta: angle between heading and velocity
    double speed_mps{};         // wheel-odometry speed fed to the last predict
};

class STATE_ESTIMATOR_API StateEstimator {
public:
    explicit StateEstimator(const EstimatorConfig& config = EstimatorConfig{});
    ~StateEstimator();

    StateEstimator(StateEstimator&&) noexcept;
    StateEstimator& operator=(StateEstimator&&) noexcept;
    StateEstimator(const StateEstimator&) = delete;
    StateEstimator& operator=(const StateEstimator&) = delete;

    const EstimatorConfig& config() const;

    // Start (or restart) the estimate at a known pose, with fresh covariance.
    void reset(double x = 0.0, double y = 0.0, double yaw_rad = 0.0);

    // One prediction tick; call it every dt_s. The inputs are exactly what the
    // car (or the DigitalTwin) gives you: the two FRONT wheel encoder speeds
    // (the undriven wheels, so they measure ground speed, not motor slip) and
    // the steering request in radians.
    EstimatedState predict(double front_left_rpm, double front_right_rpm,
                           double steering_rad);

    // Fuse an absolute pose fix (e.g. /map/pose from an external tracking
    // system). Call it whenever a NEW fix arrives - any rate works.
    void updatePose(double x, double y, double yaw_rad);

    // Fuse the IMU yaw rate (gyro z, rad/s). Typically every tick.
    void updateYawRate(double yaw_rate_radps);

    // The current fused state (valid after the first predict()).
    EstimatedState state() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cogni::state_estimator

#endif  // STATE_ESTIMATOR_STATEESTIMATOR_HPP
