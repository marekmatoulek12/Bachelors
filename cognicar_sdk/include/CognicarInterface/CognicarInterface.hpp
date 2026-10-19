#ifndef COGNICAR_INTERFACE_COGNICARINTERFACE_HPP
#define COGNICAR_INTERFACE_COGNICARINTERFACE_HPP

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <vector>

#include <CogniCommon/Export.hpp>

namespace cognicar_interface {

constexpr std::size_t kMaxLeds = 100;
constexpr auto kDefaultTelemetryMaxAge = std::chrono::milliseconds(250);

using LogMask = std::uint64_t;

constexpr LogMask kLogPoseMask = 1ULL << 0;
constexpr LogMask kLogSpeedMask = 1ULL << 1;
constexpr LogMask kLogDefaultMask = kLogPoseMask | kLogSpeedMask;
constexpr LogMask kLogFirstUserMask = 1ULL << 16;

enum class Backend {
    DirectUdp,
    Ros,
};

enum class Motor {
    FrontLeft,
    FrontRight,
    RearLeft,
    RearRight,
    Front,
    Rear,
    Body,
};

enum class LedMode : std::uint8_t {
    Off = 0,
    Car = 1,
    Rainbow = 2,
    Police = 3,
    WelcomeHomeStart = 4,
    WelcomeHomeStop = 5,
    Manual = 6,
};

enum LedControl : std::uint8_t {
    LedDaylights = 1U << 0,
    LedTail = 1U << 1,
    LedLeft = 1U << 2,
    LedRight = 1U << 3,
    LedBrake = 1U << 4,
};

struct Config {
    Backend backend{Backend::DirectUdp};
    std::string udp_ip{"192.168.0.1"};
    std::string udp_interface{"enP8p1s0"};

    std::string ros_node_name{"cognicar_interface"};
    bool spin_ros_background{true};
    // The local simulator: DirectUdp uses loopback and LiDAR localization stays off.
    bool simulation{false};
    std::string localization_config{};
    std::string localization_map{};
};

struct Rgb {
    std::uint8_t r{};
    std::uint8_t g{};
    std::uint8_t b{};
};

struct Vector3 {
    double x{};
    double y{};
    double z{};
};

struct Quaternion {
    double x{};
    double y{};
    double z{};
    double w{1.0};
};

struct Pose {
    Vector3 position{};
    Quaternion orientation{};
};

struct Twist {
    Vector3 linear{};
    Vector3 angular{};
};

struct SampleInfo {
    std::uint64_t sequence{};
    // Local steady-clock receipt time, not sensor acquisition time or ROS time.
    std::chrono::steady_clock::time_point received_at{};

    double AgeSeconds() const
    {
        return sequence == 0 ? std::numeric_limits<double>::infinity()
            : std::chrono::duration<double>(std::chrono::steady_clock::now() - received_at).count();
    }

    bool IsFresh(std::chrono::milliseconds max_age = kDefaultTelemetryMaxAge) const
    {
        return sequence != 0 && max_age.count() > 0 &&
            AgeSeconds() <= std::chrono::duration<double>(max_age).count();
    }
};

struct MotorFeedback {
    Motor motor{Motor::RearLeft};
    double speed_rpm{};
    std::int32_t torque_mNm{};
    double current_A{};
    std::uint32_t voltage_mV{};
    double temperature_C{};
    std::int16_t fault_flags{};
    SampleInfo sample{};
};

struct ServoFeedback {
    double angle_rad{};
    double rate_rads{};
    double temperature_C{};
    std::int16_t fault_flags{};
    SampleInfo sample{};
};

struct ImuData {
    Vector3 linear_acceleration_ms2{};
    Vector3 angular_velocity_rads{};
    Vector3 magnetic_field_mgauss{};
    double temperature_C{};
    std::int16_t fault_flags{};
    SampleInfo sample{};
};

struct ModelFeedback {
    std::uint8_t model_type{};
    Pose pose{};       // map frame
    Twist twist{};     // base_link / vehicle CG frame
    double beta{};     // body sideslip angle [rad]
    SampleInfo sample{};
};

struct LocalizationFeedback {
    Pose pose{};
    std::array<double, 36> covariance{};
    SampleInfo sample{};
};

struct WheelEncoderFeedback {
    Motor wheel{Motor::FrontLeft};
    double position_rad{};
    std::int32_t speed_rpm{};
    std::uint16_t fault_flags{};
    SampleInfo sample{};
};

struct LidarPoint {
    // float, not double: a scan carries thousands of these. rad, CCW from +x.
    float angle_rad{};
    std::uint16_t distance_mm{};
};

struct LidarScan {
    std::uint64_t stamp_us{};
    std::uint8_t scan_id{};
    float angle_min{};
    float angle_max{};
    float angle_increment{};
    float scan_time{};
    float range_min{};
    float range_max{};
    bool points_truncated{};
    std::vector<float> ranges{};
    std::vector<LidarPoint> points{};
    SampleInfo sample{};
};

class COGNICAR_INTERFACE_API Cognicar {
public:
    Cognicar();
    ~Cognicar();

    Cognicar(const Cognicar&) = delete;
    Cognicar& operator=(const Cognicar&) = delete;
    Cognicar(Cognicar&&) noexcept;
    Cognicar& operator=(Cognicar&&) noexcept;

    bool Initialize(const Config& config = Config{});
    void Shutdown();

    bool IsRunning() const;
    bool IsConnected(std::chrono::milliseconds max_age = kDefaultTelemetryMaxAge) const;
    void SpinOnce();

    bool SetCyclicMode(bool enabled);
    bool IsCyclicMode() const;

    bool SetRearSpeedRpm(double left_rpm, double right_rpm);
    bool SetRearCurrentA(double left_A, double right_A);
    bool SetSteeringAngleRad(double angle_rad);
    bool SetLidarEnabled(bool enabled);
    bool SetLeds(const std::vector<Rgb>& leds);
    bool SetLedMode(LedMode mode, std::uint8_t control_mask = 0, std::uint8_t control_value = 0);

    // False for missing/expired samples or a stopped backend; out is unchanged.
    bool LatestMotorFeedback(Motor motor, MotorFeedback& out,
                             std::chrono::milliseconds max_age = kDefaultTelemetryMaxAge) const;
    bool LatestMotorSpeedRequestRpm(Motor motor, double& speed_rpm) const;
    bool LatestWheelEncoder(Motor wheel, WheelEncoderFeedback& out,
                            std::chrono::milliseconds max_age = kDefaultTelemetryMaxAge) const;
    bool LatestServoFeedback(ServoFeedback& out,
                             std::chrono::milliseconds max_age = kDefaultTelemetryMaxAge) const;
    bool LatestImu(ImuData& out, std::chrono::milliseconds max_age = kDefaultTelemetryMaxAge) const;
    bool LatestModel(ModelFeedback& out, std::chrono::milliseconds max_age = kDefaultTelemetryMaxAge) const;
    // Accepted corrections from the embedded real-car localizer.
    bool LatestLocalization(LocalizationFeedback& out,
                            std::chrono::milliseconds max_age = kDefaultTelemetryMaxAge) const;
    bool LatestLidarScan(LidarScan& out, std::chrono::milliseconds max_age = kDefaultTelemetryMaxAge) const;
    bool WaitForUpdate(std::chrono::milliseconds timeout) const;

    // Starts a background thread that samples state and appends one CSV row
    // per tick at rate_hz (clamped to 500 Hz), independent of telemetry
    // arrival rate.
    bool EnableLogging(LogMask mask = kLogDefaultMask, const std::string& path = {}, double rate_hz = 100.0);
    void DisableLogging();
    bool AddLogVar(const std::string& name, LogMask mask);
    bool AddLogValue(LogMask mask, double value);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cognicar_interface

#endif // COGNICAR_INTERFACE_COGNICARINTERFACE_HPP
