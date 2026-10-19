# LiDAR localization inside CogniCar

The C++ library and Python API both run the compiled robust LiDAR localizer
internally. No separate localization process or launcher is required. This
works with ROS and DirectUdp.

```python
cfg = ci.Config()
cfg.simulation = False
car.Initialize(cfg)
car.SetLidarEnabled(True)
```

Check the return values, as the supplied autopilot does. Enabling LiDAR uses
compiled, calibrated defaults, starts the worker, and enables the sensor.
No configuration file is required or loaded by default. `Shutdown()` stops both.

In simulation, set `cfg.simulation = True` (the programs' `--sim` flag). The simulator supplies GPS and fused
model feedback; `SetLidarEnabled(True)` succeeds without starting SLAM or sending
hardware LiDAR commands.

## Run the student autopilot

From the SDK root, source `env.sh`, start the simulator/viewer as usual, then:

```bash
python3 exercises/autopilot.py --sim
```

On the real car, with the ROS bridge already running, use:

```bash
python3 exercises/autopilot.py
```

The equivalent C++ executable is `./build/autopilot [--sim]` after building the
SDK examples with CMake. The car is the default; `--sim` selects simulator GPS.
Each bundle selects its own backend. For DirectUdp hardware, set the car's
address/interface in `cfg`; `--sim` selects loopback automatically.

## Pose and readiness

Accepted scan matches supply GPS-like corrections to the car estimator
(`/gps/pose` in ROS). The tracker reads the resulting fused pose and body twist
through `LatestModel()`. After real-car LiDAR is enabled, that getter also requires
an accepted SLAM correction younger than 500 ms and a running localizer. The
example requests zero current and centres steering while inputs are missing,
and resumes when fresh inputs return. Sensor loss never falls back silently to
dead reckoning. Only accepted matches become GPS corrections.

`LatestLocalization()` gives the accepted scan pose, row-major ROS 6x6 covariance
and freshness `sample`. C++ fills an output argument and returns `bool`; Python
returns an object or `None`. Its default age limit is 250 ms. Poses use the map
frame, metres and radians. Simulation does not produce this diagnostic output.

## Map and path alignment

A new map starts at the vehicle centre, heading along +x. Start stationary with
room features in view, and use a reference path in that same coordinate frame.
The supplied oval is an example, not a map of your room.

For localization in a previously saved `.rllmap`, set `cfg.localization_map` to
its path before initialization. This selects global initialization and disables
mapping. Use a reference path recorded in that map's frame. Edit the path loaded
in the example; loading a map does not transform an unrelated path. Map creation
and saving tools remain in the standalone robust_lidar_localization package.

The SDK also includes its compiled offline `localization` Python module. Live
SLAM is linked into CogniCar; students do not need its implementation source.
