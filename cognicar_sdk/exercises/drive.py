"""Straight-line connection test. Run from the SDK root: python3 exercises/drive.py [--sim]"""
import argparse
import time

import cognicar_interface as ci

REAR_RPM = 150.0
DURATION_S = 3.0
FEEDBACK_TIMEOUT_S = 5.0
PERIOD_S = 0.01


def wheels_ready(car):
    return (car.LatestWheelEncoder(ci.Motor.FrontLeft) is not None
            and car.LatestWheelEncoder(ci.Motor.FrontRight) is not None)


def drive(car):
    deadline = time.monotonic() + FEEDBACK_TIMEOUT_S
    while not wheels_ready(car):
        if time.monotonic() >= deadline:
            return "No wheel feedback. Is the car or simulator running?"
        car.WaitForUpdate(50)

    print(f"Straight-line connection test: {REAR_RPM:g} rpm for {DURATION_S:g} s. Ctrl+C stops.")
    deadline = time.monotonic() + DURATION_S
    while time.monotonic() < deadline:
        if not wheels_ready(car):
            return "Wheel feedback lost. Stopped."
        if not car.SetSteeringAngleRad(0.0) or not car.SetRearSpeedRpm(REAR_RPM, REAR_RPM):
            return "Could not send the drive command. Stopped."
        time.sleep(PERIOD_S)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true", help="drive the simulator instead of the car")
    args = parser.parse_args()

    cfg = ci.Config()
    cfg.backend = ci.Backend.Ros
    cfg.simulation = args.sim
    car = ci.Cognicar()
    if not car.Initialize(cfg):
        raise SystemExit("Could not connect to the car or simulator.")
    try:
        error = drive(car)
    except KeyboardInterrupt:
        error = "Interrupted. Stopped."
    finally:
        car.Shutdown()
    if error:
        raise SystemExit(error)
    print("Test complete.")


if __name__ == "__main__":
    main()
