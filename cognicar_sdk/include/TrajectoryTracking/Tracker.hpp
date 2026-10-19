// Abstract tracking algorithm interface.
//
// This is the single extension point of the library: pure pursuit is the
// published reference implementation; more sophisticated controllers (e.g. MPC)
// implement the same interface and can stay private.
#ifndef COGNI_TRACKING_TRACKER_HPP
#define COGNI_TRACKING_TRACKER_HPP

#include <vector>

#include <CogniCommon/Export.hpp>
#include <TrajectoryTracking/Types.hpp>

namespace cogni::tracking {

class TRAJECTORY_TRACKING_API Tracker {
public:
    virtual ~Tracker() = default;

    // Set (or replace) the reference path. Resets internal tracking memory.
    virtual void setPath(std::vector<PathPoint> path) = 0;

    // Compute one steering command for the given vehicle pose and twist (body
    // velocity). Planar controllers use pose position x/y, the orientation yaw
    // and twist.linear.x (forward speed); twist.angular.z carries the yaw rate.
    virtual TrackingCommand compute(const Pose& pose,
                                    const Twist& twist = {}) = 0;

    // Forget per-step memory (e.g. last matched segment, previous steering).
    virtual void reset() = 0;

    // Human-readable controller name, e.g. "pure_pursuit".
    virtual const char* name() const = 0;
};

}  // namespace cogni::tracking

#endif  // COGNI_TRACKING_TRACKER_HPP
