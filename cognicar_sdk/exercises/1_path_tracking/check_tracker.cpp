// Exercise 1 check: run YOUR tracker over the path offline.
//
// No car, no simulation, no ROS. A small kinematic bicycle stands in for the
// car, so you can see whether your tracker holds the line in a second, before
// you drive anything.
//
// Build:  cmake --build build --target check_tracker
// Run:    ./build/check_tracker
#include <cmath>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <vector>

#include "student_tracker.hpp"   // the tracker you edit; also brings in the
                                 // TrajectoryTracking headers and the tt alias

namespace {

constexpr double kWheelbaseM = 0.255;   // CogniCar axle distance
constexpr double kSpeedMps = 1.0;
constexpr double kDtS = 0.01;           // same 100 Hz step the autopilot runs at
constexpr int kSteps = 4000;            // ~40 s of driving

bool fileExists(const std::string& path) { return ::access(path.c_str(), R_OK) == 0; }

// Same lookup the autopilot uses: walk up from the binary looking for a data/
// dir, so it does not matter which directory you run this from.
std::string findPathFile(const std::string& name)
{
#ifdef COGNICAR_SDK_DIR
    // Where this bundle lives, baked in by the SDK's CMakeLists.
    if (fileExists(std::string(COGNICAR_SDK_DIR) + "/data/" + name)) {
        return std::string(COGNICAR_SDK_DIR) + "/data/" + name;
    }
#endif
    char buf[4096];
    const ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    std::string dir;
    if (len > 0) { buf[len] = '\0'; dir = buf; }

    for (int up = 0; up < 5 && !dir.empty(); ++up) {
        const std::string::size_type slash = dir.find_last_of('/');
        if (slash == std::string::npos) { break; }
        dir = dir.substr(0, slash);
        const std::string candidate = dir + "/data/" + name;
        if (fileExists(candidate)) { return candidate; }
    }
    if (fileExists("data/" + name)) { return "data/" + name; }
    return {};
}

} // namespace

int main()
{
    const std::string path_file = findPathFile("oval_path_clockwise.npy");
    if (path_file.empty()) {
        std::fprintf(stderr, "could not find oval_path_clockwise.npy\n");
        return 1;
    }
    const std::vector<tt::PathPoint> path = tt::loadPath(path_file);
    StudentTracker tracker;

    // Start on the path, facing along it.
    double x = path.front().x;
    double y = path.front().y;
    double yaw = tt::projectToPath(path, x, y).heading_rad;

    double worst = 0.0;
    double sum_squared = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        const double steer_rad = tracker.step(path, tt::makePlanarPose(x, y, yaw),
                                              tt::makePlanarTwist(kSpeedMps));

        // Kinematic bicycle: what the car would do with that steering angle.
        yaw += kSpeedMps / kWheelbaseM * std::tan(steer_rad) * kDtS;
        x += kSpeedMps * std::cos(yaw) * kDtS;
        y += kSpeedMps * std::sin(yaw) * kDtS;

        const double error = std::abs(tt::projectToPath(path, x, y).cross_track_error_m);
        worst = std::max(worst, error);
        sum_squared += error * error;
    }

    std::printf("cross-track error   RMS %.3f m   worst %.3f m\n",
                std::sqrt(sum_squared / kSteps), worst);
    std::puts(worst < 0.5 ? "PASS - your tracker holds the path"
                          : "FAIL - the car leaves the path; check your steering sign and gains");
    return worst < 0.5 ? 0 : 1;
}
