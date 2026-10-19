// Donkey-and-carrot pure pursuit - student tracker (C++).
//
// This is the tracker you make your own, ONE STEP AT A TIME:
//
//     1. pick a function in the "IMPLEMENT THESE" block below
//     2. replace its body with your own math, keeping the SAME signature
//     3. run check_tracker (offline, instant), then rebuild autopilot and
//        check it still drives against the DigitalTwin
//     4. repeat until every step is yours -- then you've written the tracker
//
// Never edit the "FIXED PIPELINE" section: it always calls YOUR functions, so
// the tracker keeps running no matter how many steps you've rewritten so far.
//
// This file has no main(): it is a header #included by check_tracker.cpp (runs
// your tracker offline) and by autopilot.cpp (drives the real car). Its Python
// twin is student_tracker.py.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <TrajectoryTracking/PathIO.hpp>   // loadPath, PathPoint
#include <TrajectoryTracking/Steps.hpp>    // the reference building blocks

namespace tt = cogni::tracking;

// Tuned for the CogniCar oval track (matches config/pure_pursuit.yaml, the
// same tuning path_tracking_node runs with). The struct's own defaults are
// untuned and oscillate more; start from these instead.
inline tt::PurePursuitConfig defaultConfig()
{
    tt::PurePursuitConfig cfg;
    cfg.lookahead_distance = 0.45;       // was 0.3
    cfg.lookahead_velocity_gain = 0.35;  // was 0.15
    cfg.kp_heading = 1.6;                // was 2.5
    cfg.kp_cross_track = 0.8;            // was 1.0
    cfg.dt_s = 0.01;                     // was 0.05; must match the autopilot's 100 Hz loop
    return cfg;
}

inline tt::PurePursuitConfig CFG = defaultConfig();  // gains & limits - tweak freely


// ============================================================================
//  IMPLEMENT THESE, ONE AT A TIME.  Each starts by calling the reference tt::;
//  swap in your own body but keep the signature identical.
// ============================================================================

// Nearest point on the path -> projection (has .cross_track_error_m).
inline tt::PathProjection project_to_path(const std::vector<tt::PathPoint>& path,
                                          double x, double y)
{
    return tt::projectToPath(path, x, y);                       // TODO: your own
}

// The "carrot" a bit ahead of the car on the path -> (x, y).
inline tt::Point2 lookahead_point(const std::vector<tt::PathPoint>& path,
                                  double x, double y, double lookahead_m)
{
    return tt::lookaheadPoint(path, x, y, lookahead_m);         // TODO: your own
}

// Angle from the car's heading to the carrot, and the carrot distance.
inline tt::HeadingError heading_error(double carrot_x, double carrot_y,
                                      double x, double y, double yaw_rad)
{
    return tt::headingError(carrot_x, carrot_y, x, y, yaw_rad); // TODO: your own
}

// Raw steering angle (rad) from the errors, before clamp/slew.
inline double steering_from_errors(double heading_error_rad, double carrot_distance_m,
                                   double cross_track_error_m, const tt::PurePursuitConfig& cfg)
{
    return tt::steeringFromErrors(heading_error_rad, carrot_distance_m,
                                  cross_track_error_m, cfg);     // TODO: your own
}

// Keep the steering within the servo's physical limits.
inline double clamp_steering(double steering_rad, const tt::PurePursuitConfig& cfg)
{
    const double lo = cfg.min_steering_rad;
    const double hi = cfg.max_steering_rad;
    return std::max(lo, std::min(hi, steering_rad));            // TODO: your own
}

// Stop the steering jumping faster than the servo can move (slew limit).
inline double rate_limit(double steering_rad, double previous_steering_rad,
                         double max_rate_radps, double dt_s)
{
    return tt::rateLimit(steering_rad, previous_steering_rad,
                         max_rate_radps, dt_s);                  // TODO: your own
}


// ============================================================================
//  FIXED PIPELINE - do not edit. It wires your functions together and keeps the
//  one bit of state a real controller needs (the previous steering command, for
//  the slew limit). Call reset() when a run restarts.
// ============================================================================

class StudentTracker {
public:
    void reset() { prev_steer_rad_ = 0.0; }

    // Returns the steering command in RADIANS.
    //
    // Takes the car's full ROS-style pose (position + orientation) and twist
    // (linear + angular velocity). Today's pipeline reads x, y, yaw and the
    // forward speed - but everything else (e.g. twist.angular.z, the yaw
    // rate) arrives here too, ready when your controller wants it.
    double step(const std::vector<tt::PathPoint>& path,
                const tt::Pose& pose, const tt::Twist& twist)
    {
        const double x = pose.position.x;
        const double y = pose.position.y;
        const double yaw = tt::yawFromQuaternion(pose.orientation);
        const double speed_mps = twist.linear.x;

        const tt::PathProjection proj = project_to_path(path, x, y);

        const double lookahead =
            std::min(CFG.lookahead_distance + CFG.lookahead_velocity_gain * speed_mps,
                     CFG.max_lookahead_distance);
        const tt::Point2 carrot = lookahead_point(path, x, y, lookahead);

        const tt::HeadingError he = heading_error(carrot.x, carrot.y, x, y, yaw);
        double steer = steering_from_errors(he.error_rad, he.carrot_distance_m,
                                            proj.cross_track_error_m, CFG);

        steer = clamp_steering(steer, CFG);
        steer = rate_limit(steer, prev_steer_rad_,
                           CFG.max_steering_rate_radps, CFG.dt_s);
        prev_steer_rad_ = steer;

        return steer;
    }

private:
    double prev_steer_rad_{0.0};
};
