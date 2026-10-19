# Requirements

These are prebuilt Linux binaries. Match `BUILD_INFO.txt`; a different Python
major/minor version or Linux distribution needs a matching SDK build.

On Ubuntu 24.04, install the C++ exercise dependencies and Python runtime:

```bash
sudo apt install build-essential cmake libeigen3-dev libyaml-cpp0.8 python3 python3-numpy
```

For a bundle containing the Digital Twin, also install:

```bash
sudo apt install libqt6widgets6t64 libqt6network6t64 libqt6svg6 libfmt9
```

The ROS edition additionally needs the ROS distribution recorded in
`BUILD_INFO.txt`, including rclcpp, std_msgs, geometry_msgs, nav_msgs,
sensor_msgs and diagnostic_msgs. A ROS desktop installation includes these packages.
Install ROS first using its official Ubuntu installation guide:
https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
(use the distribution matching BUILD_INFO.txt).

Source `env.sh` in every terminal before running SDK programs.
