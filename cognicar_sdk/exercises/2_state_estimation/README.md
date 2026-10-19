# Exercise 2: state estimation

**Edit** `student_kalman.py` or `student_kalman.hpp`: a Kalman filter for the
car's pose `[x, y, yaw]`. It predicts with the wheel speed and the gyro and
corrects with the pose fix. Replace one `se.*` call at a time and keep the
function signatures.

**Check** offline, no car or simulator needed. It must print **PASS**: your
filter beats the raw pose fix and survives a 3 s fix dropout.

```bash
python3 exercises/2_state_estimation/check_kalman.py
```

C++: `cmake --build build --target check_kalman -j2`, then `./build/check_kalman`.

**Run** the autopilot: it drives on your filter's pose.
