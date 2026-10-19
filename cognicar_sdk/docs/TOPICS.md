# CogniCar ROS topic contract

This is the language- and OS-independent interface between a control application
and the car (or a simulation). Any ROS 2 node on any platform that speaks these
topics can drive and observe the vehicle — the compiled SDK is just a convenience
wrapper around them.

On the real car the `ros_concentrator` bridge subscribes to the command topics
and publishes the feedback topics (it relays them to/from the MCU over UDP). A
simulation (e.g. the DigitalTwin package) plays the same role: it must subscribe
to the commands and publish the feedback.

All topics use the default ROS 2 QoS unless noted. Use `ROS_DOMAIN_ID` and your
DDS configuration to place the control node and the car/sim on the same graph.

## Commands — control node publishes, car/sim subscribes

| Topic | Type | Meaning |
|-------|------|---------|
| `/powertrain/rl/rpm_req` | `std_msgs/Float64` | rear-left wheel speed request (rpm) |
| `/powertrain/rr/rpm_req` | `std_msgs/Float64` | rear-right wheel speed request (rpm) |
| `/powertrain/rl/current_req` | `std_msgs/Float64` | rear-left motor current request (A) |
| `/powertrain/rr/current_req` | `std_msgs/Float64` | rear-right motor current request (A) |
| `/servo/front/angle_req` | `std_msgs/Float64` | front steering angle request (rad) |
| `/gps/pose` | `geometry_msgs/PoseStamped` | externally supplied body pose |
| `/led_cmd` | `std_msgs/UInt8MultiArray` | indexed LEDs: `[count, r,g,b, r,g,b, ...]` |
| `/led_mode_cmd` | `std_msgs/UInt8MultiArray` | `[mode, control_mask, control_value]` |
| `/lidar/command` | `std_msgs/UInt8` | LIDAR enable (0/1). QoS: reliable, transient-local |

The two `rpm_req` topics are the exception to the one-way rule: the interface
also subscribes to them, so `LatestMotorSpeedRequestRpm()` reads back the
commanded wheel speed no matter which node published it. That readback is the
only reason it listens - it never treats another node's request as its own
command, and its own "is the car listening?" check discounts that subscription.

## Feedback — car/sim publishes, control node subscribes

| Topic | Type | Meaning |
|-------|------|---------|
| `/powertrain/rl/rpm_fb` | `std_msgs/Float64` | rear-left wheel speed (rpm) |
| `/powertrain/rr/rpm_fb` | `std_msgs/Float64` | rear-right wheel speed (rpm) |
| `/powertrain/rl/current_fb` | `std_msgs/Float64` | rear-left motor current (A) |
| `/powertrain/rr/current_fb` | `std_msgs/Float64` | rear-right motor current (A) |
| `/powertrain/rl/status` | `std_msgs/Int16` | rear-left fault flags |
| `/powertrain/rr/status` | `std_msgs/Int16` | rear-right fault flags |
| `/powertrain/rl/temp` | `std_msgs/Float64` | rear-left motor temperature (C) |
| `/powertrain/rr/temp` | `std_msgs/Float64` | rear-right motor temperature (C) |
| `/servo/front/angle_fb` | `std_msgs/Float64` | front steering angle (rad) |
| `/powertrain/fl/wheel_enc` | `std_msgs/Float64` | front-left wheel encoder speed (rpm) |
| `/powertrain/fr/wheel_enc` | `std_msgs/Float64` | front-right wheel encoder speed (rpm) |
| `/imu` | `sensor_msgs/Imu` | IMU linear acceleration + angular velocity |
| `/state/odometry` | `nav_msgs/Odometry` | coherent model state: pose in `map`, twist in `base_link`; default frames are configurable with `map_frame_id`/`base_frame_id` |
| `/map/pose` | `geometry_msgs/Pose` | model pose in the map frame |
| `/map/twist` | `geometry_msgs/Twist` | legacy unheadered model twist in the vehicle CG/body frame (`base_link`) |
| `/map/side_slip_angle` | `std_msgs/Float64` | body sideslip angle (beta), rad |
| `/scan` | `sensor_msgs/LaserScan` | LIDAR scan, 720 angular bins. QoS: SensorData (best-effort) |
| `/lidar/points` | `sensor_msgs/PointCloud2` | full LIDAR cloud: every return (up to 1440 pts), XYZ in the `lidar` frame (z=0). QoS: SensorData |

## Implementing a simulation against this contract

A simulator is a drop-in stand-in for the car if it:
1. subscribes to every **command** topic and actuates its vehicle model, and
2. publishes the **feedback** topics your control node reads
   (at minimum `/state/odometry`, the legacy `/map/pose` and `/map/twist`,
   the wheel/steering feedback, and
   `/scan` if you use LIDAR; publish `/lidar/points` too for the full cloud).

`/state/odometry` is the preferred state interface. Its header uses the bridge
publication time; model covariance is not present on the current UDP contract,
so the covariance arrays are zero-filled. The two legacy `/map/*` messages are
published from the same model event for backward compatibility.

Steering commands and feedback use radians throughout ROS, the SDK and UDP v8.
Update all ROS steering producers and consumers together; topic names and
message types have not changed. Recordings made with the old degree-based ROS
contract must be converted before replay through these interfaces.

## Embedded LiDAR localization

`SetLidarEnabled(true)` starts the compiled localizer inside CogniCar on the
real car. It consumes scan, IMU and wheel feedback, and publishes accepted
corrections to `/gps/pose` for the car estimator. `LatestLocalization()` reads
these internal accepted poses; `LatestModel()` reads the fused car state and
requires a correction within 500 ms while localization is enabled. There is
no separate localization node or measurement topic in this SDK workflow.
`Config.simulation=true` uses simulator GPS and does not start SLAM.
