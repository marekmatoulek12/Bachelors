// Reference-path geometry: nearest-point projection, arc-length sampling and
// lookahead lookup. This is a faithful C++ port of the Python
// trajectory_tracking_core.base projection logic, kept separate from any
// control law so the geometry can be taught and reused on its own.
#ifndef COGNI_TRACKING_REFERENCE_PATH_HPP
#define COGNI_TRACKING_REFERENCE_PATH_HPP

#include <cstddef>
#include <optional>
#include <vector>

#include <CogniCommon/Export.hpp>
#include <TrajectoryTracking/Types.hpp>

namespace cogni::tracking {

struct PathConfig {
    bool loop_trajectory{true};      // treat the path as a closed loop
    int segment_search_window{20};   // segments searched around the last match
    double reset_distance_m{3.0};    // if nearest match is farther, re-search all
};

class TRAJECTORY_TRACKING_API ReferencePath {
public:
    ReferencePath() = default;
    explicit ReferencePath(PathConfig config) : config_(config) {}

    void setConfig(PathConfig config) { config_ = config; }
    const PathConfig& config() const { return config_; }

    // Replace the path points and reset the segment cache. Points are stored
    // as-is (already normalized: x, y, heading, speed).
    void setPoints(std::vector<PathPoint> points);

    bool empty() const { return points_.empty(); }
    std::size_t size() const { return points_.size(); }
    const std::vector<PathPoint>& points() const { return points_; }

    void reset() { last_segment_index_.reset(); }

    // Project the vehicle position onto the nearest path segment.
    PathProjection project(const VehicleState& state,
                           bool update_index = true,
                           bool search_all = false);

    // Sample the path at a given arc length (wraps when loop_trajectory).
    PathPoint sampleAt(double progress_m) const;

    // Lookahead "carrot": projection arc length plus lookahead_distance.
    Point2 lookaheadPoint(const VehicleState& state,
                          double lookahead_distance,
                          double* out_heading_rad = nullptr);

    static double normalizeAngle(double angle);

private:
    struct Segments {
        std::vector<Point2> start;
        std::vector<Point2> vec;     // end - start
        std::vector<double> length;
        std::vector<double> cum_len; // size N+1, prefix arc lengths
    };

    Segments buildSegments() const;
    double curvatureAt(const Segments& seg, std::size_t seg_idx) const;
    double headingAt(std::size_t seg_idx, double ratio,
                     const Point2& seg_vec, double seg_len) const;
    double speedAt(std::size_t seg_idx, double ratio) const;

    PathConfig config_{};
    std::vector<PathPoint> points_{};
    std::optional<std::size_t> last_segment_index_{};
};

}  // namespace cogni::tracking

#endif  // COGNI_TRACKING_REFERENCE_PATH_HPP
