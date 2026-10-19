"""Exercise 1 check: run YOUR tracker over the path offline.

No car, no simulation, no ROS. A small kinematic bicycle stands in for the car,
so you can see whether your tracker holds the line in a second, before you
drive anything.

    python3 check_tracker.py
"""
import math
import os
import sys

import cogni_tracking as tt

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from student_tracker import StudentTracker  # noqa: E402  (this is YOUR tracker)

WHEELBASE_M = 0.255   # CogniCar axle distance
SPEED_MPS = 1.0
DT_S = 0.01           # same 100 Hz step the autopilot runs at
LAPS_STEPS = 4000     # ~40 s of driving


def find_path_file(name):
    """Same lookup the autopilot uses: resolve from this file, not the CWD."""
    candidates = []
    directory = _here
    for _ in range(5):
        candidates.append(os.path.join(directory, "data", name))
        directory = os.path.dirname(directory)
    candidates.append(os.path.join(_here, "..", "..", "TrajectoryTracking", "data", name))
    candidates.append(os.path.join(os.getcwd(), "data", name))
    for candidate in candidates:
        resolved = os.path.normpath(candidate)
        if os.path.isfile(resolved):
            return resolved
    raise SystemExit("could not find " + name)


def main():
    path = tt.load_path(find_path_file("oval_path_clockwise.npy"))
    tracker = StudentTracker()

    # Start on the path, facing along it.
    x, y = float(path[0][0]), float(path[0][1])
    yaw = tt.project_to_path(path, x, y).heading_rad

    worst = 0.0
    sum_squared = 0.0
    for _ in range(LAPS_STEPS):
        steer_rad = tracker.step(path, tt.make_planar_pose(x, y, yaw),
                                 tt.make_planar_twist(SPEED_MPS))

        # Kinematic bicycle: what the car would do with that steering angle.
        yaw += SPEED_MPS / WHEELBASE_M * math.tan(steer_rad) * DT_S
        x += SPEED_MPS * math.cos(yaw) * DT_S
        y += SPEED_MPS * math.sin(yaw) * DT_S

        error = abs(tt.project_to_path(path, x, y).cross_track_error_m)
        worst = max(worst, error)
        sum_squared += error * error

    rms = math.sqrt(sum_squared / LAPS_STEPS)
    print("cross-track error   RMS {:.3f} m   worst {:.3f} m".format(rms, worst))
    print("PASS - your tracker holds the path" if worst < 0.5 else
          "FAIL - the car leaves the path; check your steering sign and gains")

    return 0 if worst < 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
