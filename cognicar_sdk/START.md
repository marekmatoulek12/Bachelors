# First run

Install the [requirements](docs/REQUIREMENTS.md) once. Then run the autopilot:
it drives the car around the oval path.

Power the car on and connect this PC to its network. Then:

```bash
cd cognicar_sdk
source env.sh
python3 exercises/autopilot.py
```

C++ instead of Python:

```bash
cmake -S . -B build && cmake --build build --target autopilot -j2
./build/autopilot
```

It prints `Driving.` once the car has localized itself. Ctrl+C stops it.
Read [localization](docs/LOCALIZATION.md) before driving. Without ROS, first set
`cfg.udp_ip` and `cfg.udp_interface` in the program.

Next, make the autopilot yours: [exercises](exercises/README.md).
Guides: [Getting started](docs/getting-started-handout.pdf),
[Path tracking](docs/pathtracking-handout.pdf), [API](docs/cognicar-api.pdf).
