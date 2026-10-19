// Follow the reference path. Run from the SDK root: ./build/autopilot [--sim]
#include <CognicarInterface/CognicarInterface.hpp>
#include "student_kalman.hpp"
#include "student_tracker.hpp"
#include "student_low_level_control.hpp"
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

namespace ci = cognicar_interface;

constexpr char kPathFile[] = "data/oval_path_clockwise.npy";
constexpr double kTargetSpeedMps = 1.0;
constexpr std::chrono::milliseconds kPeriod{10};
constexpr double kPeriodS = std::chrono::duration<double>(kPeriod).count();

volatile std::sig_atomic_t stop_requested = 0;
void requestStop(int) { stop_requested = 1; }

int main(int argc, char** argv)
{
    const bool simulation = argc == 2 && std::string(argv[1]) == "--sim";
    if (argc > 1 && !simulation) {
        std::fputs("Usage: autopilot [--sim]\n", stderr);
        return 2;
    }
    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);

    const auto path = tt::loadPath(kPathFile);
    StudentKalman kalman;
    StudentTracker tracker;
    StudentLowLevelControl controller;
    ci::Config cfg;
    cfg.backend = ci::Backend::Ros;
    cfg.simulation = simulation;
    ci::Cognicar car;
    if (!car.Initialize(cfg)) {
        std::fputs("Could not connect to the car or simulator.\n", stderr);
        return 1;
    }
    if (!car.SetLidarEnabled(true)) {
        std::fputs("Could not start LiDAR localization.\n", stderr);
        car.Shutdown();
        return 1;
    }

    std::puts("Waiting for feedback.");
    bool localized = false;
    std::uint64_t last_fix = 0;
    bool driving = false;
    int result = 0;
    auto next_tick = std::chrono::steady_clock::now();
    while (!stop_requested) {
        next_tick += kPeriod;
        // The pose fix: LiDAR localization on the car, the simulator's own pose in --sim.
        ci::Pose fix;
        std::uint64_t fix_sequence = 0;
        if (simulation) {
            ci::ModelFeedback model;
            if (car.LatestModel(model)) { fix = model.pose; fix_sequence = model.sample.sequence; }
        } else {
            ci::LocalizationFeedback localization;
            if (car.LatestLocalization(localization, std::chrono::seconds(1))) {
                fix = localization.pose;
                fix_sequence = localization.sample.sequence;
            }
        }
        if (fix_sequence != 0 && fix_sequence != last_fix) {
            last_fix = fix_sequence;
            const double fix_yaw = tt::yawFromQuaternion(
                {fix.orientation.x, fix.orientation.y, fix.orientation.z, fix.orientation.w});
            if (localized) {
                kalman.update(fix.position.x, fix.position.y, fix_yaw);
            } else {
                kalman.reset(fix.position.x, fix.position.y, fix_yaw);
                localized = true;
            }
        }

        ci::ImuData imu;
        ci::WheelEncoderFeedback left, right;
        const bool ready = localized && fix_sequence != 0 && car.LatestImu(imu)
            && car.LatestWheelEncoder(ci::Motor::FrontLeft, left)
            && car.LatestWheelEncoder(ci::Motor::FrontRight, right);
        if (ready != driving) {
            std::puts(ready ? "Driving." : "Feedback lost. Car stopped, waiting.");
            driving = ready;
        }
        if (ready) {
            kalman.predict(left.speed_rpm, right.speed_rpm, imu.angular_velocity_rads.z);
            const tt::Pose pose = tt::makePlanarPose(kalman.x(), kalman.y(), kalman.yaw());
            const tt::Twist twist = tt::makePlanarTwist(kalman.speed());
            const double steering = tracker.step(path, pose, twist);
            const auto currents = controller.step(steering, kTargetSpeedMps, left.speed_rpm, right.speed_rpm,
                                                  imu.angular_velocity_rads.z, kPeriodS);
            if (!car.SetSteeringAngleRad(steering) || !car.SetRearCurrentA(currents[0], currents[1])) {
                std::fputs("Could not send the drive command.\n", stderr);
                result = 1;
                break;
            }
        } else {
            car.SetRearCurrentA(0.0, 0.0);
            car.SetSteeringAngleRad(0.0);
            localized = false;
            tracker.reset();
            controller.reset();
        }
        std::this_thread::sleep_until(next_tick);
    }
    car.Shutdown();
    std::puts("Stopped.");
    return result;
}
