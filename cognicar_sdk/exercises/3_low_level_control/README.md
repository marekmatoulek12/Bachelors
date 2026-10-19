# Exercise 3: low-level control

**Edit** `student_low_level_control.py` or `student_low_level_control.hpp`.
Replace one `llc.*` call at a time and keep the function signatures.

**Check** offline, no car or simulator needed. It must print **PASS**:

```bash
python3 exercises/3_low_level_control/check_low_level_control.py
```

C++: `cmake --build build --target check_low_level_control -j2`, then
`./build/check_low_level_control`.

**Run** your controller in the autopilot:

```bash
python3 exercises/autopilot.py --sim
```

C++: `./build/autopilot --sim` after rebuilding the `autopilot` target.
The car should hold 1 m/s around the oval.
