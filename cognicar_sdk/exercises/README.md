# Exercises

Run every command from the SDK root after `source env.sh`. Programs that drive
take `--sim` for the simulator; without it they drive the real car.
Start the simulator as in [START.md](../START.md).

| Program or folder | What it is |
| --- | --- |
| `drive.py` / `.cpp` | Straight-line connection test: 150 rpm for 3 s. |
| `autopilot.py` / `.cpp` | Follows the oval with your filter, tracker and low-level controller. |
| `0_hello_car/` | First API calls, one TODO at a time. |
| `1_path_tracking/` | Your steering: `student_tracker`. |
| `2_state_estimation/` | Your pose filter: `student_kalman`. |
| `3_low_level_control/` | Your speed and torque control: `student_low_level_control`. |

In each exercise, replace one reference call at a time and rerun its checker.
The autopilot prints `Waiting for feedback.`, `Driving.` or
`Feedback lost. Car stopped, waiting.` whenever its state changes.

C++: `cmake -S . -B build && cmake --build build -j2` builds every program into
`build/`. Rebuild after each edit.
