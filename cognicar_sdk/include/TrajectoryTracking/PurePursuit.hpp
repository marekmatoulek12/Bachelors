// Pure-pursuit tracker: classic "donkey-and-carrot" steering plus an explicit
// cross-track P term and a steering-rate limit. Faithful C++ port of the Python
// trajectory_tracking_core.pure_pursuit, with identical default gains.
#ifndef COGNI_TRACKING_PURE_PURSUIT_HPP
#define COGNI_TRACKING_PURE_PURSUIT_HPP

#include <vector>

#include <CogniCommon/Export.hpp>
#include <TrajectoryTracking/ReferencePath.hpp>
#include <TrajectoryTracking/Tracker.hpp>
#include <TrajectoryTracking/Types.hpp>

namespace cogni::tracking {

struct PurePursuitConfig {
    // Lookahead grows with speed, then is capped.
    double lookahead_distance{0.3};
    double lookahead_velocity_gain{0.15};
    double max_lookahead_distance{1.2};

    double kp_heading{2.5};        // gain on the pure-pursuit steering angle
    double wheelbase_m{0.255};
    double kp_cross_track{1.0};    // explicit cross-track correction gain

    // See yawDamping in Steps.hpp. 0 disables it.
    double kd_yaw_rate{0.0};               // [s] steering per rad/s of excess yaw
    double yaw_damper_min_speed_mps{0.3};  // below this the measured rate is noise

    double max_steering_rate_radps{5.0};  // steering slew limit
    double dt_s{0.05};                     // period used for the slew limit

    double min_steering_rad{-0.873};   // rad (~-50 deg)
    double max_steering_rad{0.873};    // rad (~50 deg)

    PathConfig path{};  // loop / search window / reset distance
};

class TRAJECTORY_TRACKING_API PurePursuit final : public Tracker {
public:
    PurePursuit() : PurePursuit(PurePursuitConfig{}) {}
    explicit PurePursuit(const PurePursuitConfig& config);

    void setConfig(const PurePursuitConfig& config);
    const PurePursuitConfig& config() const { return config_; }

    void setPath(std::vector<PathPoint> path) override;
    TrackingCommand compute(const Pose& pose, const Twist& twist = {}) override;
    void reset() override;
    const char* name() const override { return "pure_pursuit"; }

    double minSteeringRad() const { return min_steer_rad_; }
    double maxSteeringRad() const { return max_steer_rad_; }

private:
    PurePursuitConfig config_{};
    ReferencePath path_{};
    double min_steer_rad_{};
    double max_steer_rad_{};
    double previous_steering_rad_{};
};

}  // namespace cogni::tracking

#endif  // COGNI_TRACKING_PURE_PURSUIT_HPP
