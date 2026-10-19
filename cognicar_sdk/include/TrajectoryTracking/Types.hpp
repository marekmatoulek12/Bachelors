// Public, ROS-independent value types for the trajectory tracking core.
//
// The tracker boundary (Tracker::compute) speaks ROS-style 3D geometry: a Pose
// (position + orientation quaternion) and a Twist (linear + angular velocity),
// laid out exactly like geometry_msgs/Pose and geometry_msgs/Twist so they map
// 1:1 at the ROS node. Today's planar controllers only read x/y/yaw and the
// forward speed, but keeping the interface 3D means a future controller (or a
// Z / full-3D-rotation upgrade) needs no interface change -- trackers stay
// swappable behind the same signature. The internal control state stays the
// scalar VehicleState below.
#ifndef COGNI_TRACKING_TYPES_HPP
#define COGNI_TRACKING_TYPES_HPP

#include <cmath>
#include <optional>
#include <string>

namespace cogni::tracking {

// 3D vector (geometry_msgs/Vector3 / Point layout).
struct Vector3 {
    double x{};
    double y{};
    double z{};
};

// Unit quaternion (geometry_msgs/Quaternion layout). Identity (w=1) faces +x.
struct Quaternion {
    double x{};
    double y{};
    double z{};
    double w{1.0};
};

// Rigid-body pose: position + orientation (geometry_msgs/Pose layout).
struct Pose {
    Vector3 position{};
    Quaternion orientation{};
};

// Body velocity: linear + angular (geometry_msgs/Twist layout). For a planar
// vehicle, linear.x is the forward speed and angular.z is the yaw rate.
struct Twist {
    Vector3 linear{};
    Vector3 angular{};
};

// Yaw (rotation about +z) extracted from a quaternion.
inline double yawFromQuaternion(const Quaternion& q)
{
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

// Heading in RADIANS from a quaternion's components (x, y, z, w). Convenience
// for turning a vehicle pose (e.g. the CognicarInterface telemetry quaternion)
// into the yaw the planar trackers expect.
inline double yawRad(double x, double y, double z, double w)
{
    return yawFromQuaternion(Quaternion{x, y, z, w});
}

// Quaternion for a planar yaw-only rotation about +z.
inline Quaternion quaternionFromYaw(double yaw_rad)
{
    Quaternion q;
    q.z = std::sin(0.5 * yaw_rad);
    q.w = std::cos(0.5 * yaw_rad);
    return q;
}

// Convenience constructors for the common planar case.
inline Pose makePlanarPose(double x, double y, double yaw_rad)
{
    return Pose{Vector3{x, y, 0.0}, quaternionFromYaw(yaw_rad)};
}

inline Twist makePlanarTwist(double speed_mps, double yaw_rate_radps = 0.0)
{
    Twist twist;
    twist.linear.x = speed_mps;
    twist.angular.z = yaw_rate_radps;
    return twist;
}

// Internal controller input state in the map frame (scalar / planar).
struct VehicleState {
    double x{};
    double y{};
    double yaw_rad{};
    double speed_mps{};
    double yaw_rate_radps{};
};

// One reference-path sample: position, travel heading and desired speed.
struct PathPoint {
    double x{};
    double y{};
    double heading_rad{};
    double speed_mps{};
};

struct Point2 {
    double x{};
    double y{};
};

// Nearest projection of the vehicle onto the reference path, plus the usual
// tracking errors derived from it.
struct PathProjection {
    Point2 point{};
    double heading_rad{};
    double speed_mps{};
    int segment_index{};
    double progress_m{};
    double cross_track_error_m{};  // signed: >0 means vehicle is left of the path
    double distance_m{};
    // Local path curvature [1/m], signed like the heading change (>0 = left
    // turn). speed * curvature is the yaw rate the path asks for at this point,
    // which is what the yaw damper measures the car against.
    double curvature_1pm{};
};

// Result of one tracking step.
struct TrackingCommand {
    double steering_rad{};
    std::string controller{};
    std::optional<Point2> target_point{};  // the lookahead "carrot", if any
    double target_heading_rad{};
    double target_speed_mps{};
    double heading_error_rad{};
    double cross_track_error_m{};
};

}  // namespace cogni::tracking

#endif  // COGNI_TRACKING_TYPES_HPP
