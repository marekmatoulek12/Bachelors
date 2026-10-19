// Straight-line connection test. Run from the SDK root: ./build/drive [--sim]
#include <CognicarInterface/CognicarInterface.hpp>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>

namespace ci = cognicar_interface;
using namespace std::chrono_literals;

constexpr double kRearRpm = 150.0;
constexpr auto kDuration = 3s;
constexpr auto kFeedbackTimeout = 5s;
constexpr auto kPeriod = 10ms;

volatile std::sig_atomic_t stop_requested = 0;
void requestStop(int) { stop_requested = 1; }

bool wheelsReady(const ci::Cognicar& car)
{
    ci::WheelEncoderFeedback left, right;
    return car.LatestWheelEncoder(ci::Motor::FrontLeft, left)
        && car.LatestWheelEncoder(ci::Motor::FrontRight, right);
}

// Returns an error message, or nullptr when the test completed.
const char* drive(ci::Cognicar& car)
{
    const auto ready_deadline = std::chrono::steady_clock::now() + kFeedbackTimeout;
    while (!wheelsReady(car)) {
        if (stop_requested) return "Interrupted. Stopped.";
        if (std::chrono::steady_clock::now() >= ready_deadline) {
            return "No wheel feedback. Is the car or simulator running?";
        }
        car.WaitForUpdate(50ms);
    }

    std::printf("Straight-line connection test: %.0f rpm for 3 s. Ctrl+C stops.\n", kRearRpm);
    const auto deadline = std::chrono::steady_clock::now() + kDuration;
    while (std::chrono::steady_clock::now() < deadline) {
        if (stop_requested) return "Interrupted. Stopped.";
        if (!wheelsReady(car)) return "Wheel feedback lost. Stopped.";
        if (!car.SetSteeringAngleRad(0.0) || !car.SetRearSpeedRpm(kRearRpm, kRearRpm)) {
            return "Could not send the drive command. Stopped.";
        }
        std::this_thread::sleep_for(kPeriod);
    }
    return nullptr;
}

int main(int argc, char** argv)
{
    const bool simulation = argc == 2 && std::string(argv[1]) == "--sim";
    if (argc > 1 && !simulation) {
        std::fputs("Usage: drive [--sim]\n", stderr);
        return 2;
    }
    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);

    ci::Config cfg;
    cfg.backend = ci::Backend::Ros;
    cfg.simulation = simulation;
    ci::Cognicar car;
    if (!car.Initialize(cfg)) {
        std::fputs("Could not connect to the car or simulator.\n", stderr);
        return 1;
    }
    const char* error = drive(car);
    car.Shutdown();
    if (error) {
        std::fprintf(stderr, "%s\n", error);
        return 1;
    }
    std::puts("Test complete.");
}
