# Exercise 1: path tracking

**Edit** `student_tracker.py` or `student_tracker.hpp`. Replace one `tt.*` call
at a time and keep the function signatures.

**Check** offline, no car or simulator needed. It must print **PASS**:

```bash
python3 exercises/1_path_tracking/check_tracker.py
```

C++: `cmake --build build --target check_tracker -j2`, then `./build/check_tracker`.

**Run** your tracker in the autopilot:

```bash
python3 exercises/autopilot.py --sim
```

C++: `./build/autopilot --sim` after rebuilding the `autopilot` target.
The car should follow the oval. The [Path tracking guide](../../docs/pathtracking-handout.pdf)
explains the geometry and tuning.
