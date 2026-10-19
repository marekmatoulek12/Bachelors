// Exercise 0 - hello car. Run from the SDK root: ./build/hello_car [--sim]
//
// Fill in the TODOs top to bottom, build, run, watch the car. One call per step.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>

#include <CognicarInterface/CognicarInterface.hpp>

namespace ci = cognicar_interface;
using namespace std::chrono_literals;

constexpr char kLogPath[] = "hello_car_log.csv";
constexpr ci::LogMask kCurrentColumn = ci::kLogFirstUserMask;  // one bit per column

// Did the logger write what you asked for?
void checkLog()
{
    std::ifstream log(kLogPath);
    std::string header, line;
    if (!std::getline(log, header)) { std::puts("no log file"); return; }
    int rows = 0;
    while (std::getline(log, line)) { ++rows; }
    std::printf("%s: %d rows\ncolumns: %s\n", kLogPath, rows, header.c_str());
}

int main(int argc, char** argv)
{
    const bool simulation = argc == 2 && std::string(argv[1]) == "--sim";
    if (argc > 1 && !simulation) {
        std::fputs("Usage: hello_car [--sim]\n", stderr);
        return 2;
    }

    ci::Cognicar car;
    ci::Config cfg;
    cfg.backend = ci::Backend::Ros;
    cfg.simulation = simulation;
    // Without ROS, straight to the car over UDP:
    // cfg.backend = ci::Backend::DirectUdp;
    // cfg.udp_ip = "192.168.0.1";
    // cfg.udp_interface = "enP8p1s0";
    if (!car.Initialize(cfg)) {
        std::fputs("Could not connect to the car or simulator.\n", stderr);
        return 1;
    }

    car.SetCyclicMode(true);  // resend commands, so the car does not time out

    // TODO 1  Lights on. Try Car, Rainbow, Police, WelcomeHomeStart, Off.
    car.SetLedMode(ci::LedMode::Car);
    std::this_thread::sleep_for(2s);

    // TODO 2  Log pose and speed, plus a column of your own. Register the
    //         column BEFORE EnableLogging - after it, registration is locked.
    // car.AddLogVar("motor_current_A", kCurrentColumn);
    // car.EnableLogging(ci::kLogDefaultMask | kCurrentColumn, kLogPath);

    // TODO 3  Steer. Radians, positive is left, stay within +-0.35.
    // car.SetSteeringAngleRad(0.0);
    std::this_thread::sleep_for(1s);

    // TODO 4  Start the rear motors, left and right, at 300 rpm.
    // car.SetRearSpeedRpm(0.0, 0.0);

    // TODO 5  Drive for 3 s: read the rear left motor and fill your column.
    for (int i = 0; i < 30; ++i) {
        // ci::MotorFeedback fb;
        // if (car.LatestMotorFeedback(ci::Motor::RearLeft, fb)) {
        //     std::printf("%.1f rpm  %.2f A  %.1f C\n", fb.speed_rpm, fb.current_A, fb.temperature_C);
        //     car.AddLogValue(kCurrentColumn, fb.current_A);
        // }
        std::this_thread::sleep_for(100ms);
    }

    // TODO 6  Stop safely: motors to 0, wheels straight, lights off.

    car.DisableLogging();
    car.Shutdown();

    // TODO 7  Check your work - uncomment once TODO 2 logs something.
    // checkLog();
    return 0;
}
