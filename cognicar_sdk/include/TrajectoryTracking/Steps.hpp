// Stateless building blocks of the pure-pursuit pipeline. These are the exact
// functions PurePursuit::compute calls, and the reference units students
// replace one at a time from Python.
#ifndef COGNI_TRACKING_STEPS_HPP
#define COGNI_TRACKING_STEPS_HPP

#include <vector>

#include <CogniCommon/Export.hpp>          // TRAJECTORY_TRACKING_API
#include <TrajectoryTracking/PurePursuit.hpp>     // PurePursuitConfig
#include <TrajectoryTracking/ReferencePath.hpp>   // PathConfig
#include <TrajectoryTracking/Types.hpp>           // PathPoint, Point2, PathProjection

namespace cogni::tracking {

// Heading error to the carrot, plus the (floored) carrot distance.
struct HeadingError {
    double error_rad{};
    double carrot_distance_m{};
};
TRAJECTORY_TRACKING_API HeadingError headingError(double carrot_x, double carrot_y,
                                                  double x, double y, double yaw_rad);

// Pure-pursuit + explicit cross-track steering law (pre clamp / rate-limit).
TRAJECTORY_TRACKING_API double steeringFromErrors(double heading_error_rad, double carrot_distance_m,
                                                  double cross_track_error_m, const PurePursuitConfig& cfg);

// Damps the yaw rate in excess of what the path curvature calls for:
//
//     steering -= kd_yaw_rate * (yaw_rate_measured - speed * path_curvature)
//
// Zero in a steady corner (the car's yaw rate matches the curvature); nonzero
// on corner exit, where curvature drops to zero but the car still carries
// yaw - that excess is the overshoot this kills. No-op when kd_yaw_rate <= 0
// or speed is below yaw_damper_min_speed_mps.
//
// yaw_rate_radps must share the pose yaw's sign convention (REP-103: +z up,
// CCW positive). A flipped sign makes this positive feedback - verify before
// raising the gain.
TRAJECTORY_TRACKING_API double yawDamping(double steering_rad, double yaw_rate_radps,
                                          double yaw_rate_reference_radps, double speed_mps,
                                          const PurePursuitConfig& cfg);

// Steering slew limit: bound the change from the previous command to
// max_rate_radps * dt_s. Pure pursuit's final, stateful step (the caller holds
// the previous command); returns steering_rad unchanged if the budget is <= 0.
TRAJECTORY_TRACKING_API double rateLimit(double steering_rad, double previous_steering_rad,
                                         double max_rate_radps, double dt_s);

// Stateless full-path projection (searches the whole path; no segment cache).
TRAJECTORY_TRACKING_API PathProjection projectToPath(const std::vector<PathPoint>& path,
                                                     double x, double y, const PathConfig& cfg = {});

// Stateless lookahead "carrot": project, then sample at progress + lookahead.
TRAJECTORY_TRACKING_API Point2 lookaheadPoint(const std::vector<PathPoint>& path, double x, double y,
                                              double lookahead_m, const PathConfig& cfg = {});

}  // namespace cogni::tracking

#endif  // COGNI_TRACKING_STEPS_HPP
