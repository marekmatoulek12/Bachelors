"""Exercise 0 - hello car. Run from the SDK root: python3 exercises/0_hello_car/hello_car.py [--sim]

Fill in the TODOs top to bottom, run, watch the car. One call per step.
"""
import argparse
import time

import cognicar_interface as ci

LOG_PATH = "hello_car_log.csv"
CURRENT_COLUMN = ci.kLogFirstUserMask  # one mask bit per column of your own


def check_log():
    """Did the logger write what you asked for?"""
    with open(LOG_PATH) as f:
        rows = f.read().splitlines()
    print("{}: {} rows".format(LOG_PATH, len(rows) - 1))
    print("columns: {}".format(rows[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true", help="drive the simulator instead of the car")
    args = parser.parse_args()

    car = ci.Cognicar()
    cfg = ci.Config()
    cfg.backend = ci.Backend.Ros
    cfg.simulation = args.sim
    # Without ROS, straight to the car over UDP:
    # cfg.backend = ci.Backend.DirectUdp
    # cfg.udp_ip = "192.168.0.1"
    # cfg.udp_interface = "enP8p1s0"
    if not car.Initialize(cfg):
        raise SystemExit("Could not connect to the car or simulator.")

    car.SetCyclicMode(True)  # resend commands, so the car does not time out

    # TODO 1  Lights on. Try Car, Rainbow, Police, WelcomeHomeStart, Off.
    car.SetLedMode(ci.LedMode.Car)
    time.sleep(2.0)

    # TODO 2  Log pose and speed, plus a column of your own. Register the
    #         column BEFORE EnableLogging - after it, registration is locked.
    # car.AddLogVar("motor_current_A", CURRENT_COLUMN)
    # car.EnableLogging(ci.kLogDefaultMask | CURRENT_COLUMN, LOG_PATH)

    # TODO 3  Steer. Radians, positive is left, stay within +-0.35.
    # car.SetSteeringAngleRad(0.0)
    time.sleep(1.0)

    # TODO 4  Start the rear motors, left and right, at 300 rpm.
    # car.SetRearSpeedRpm(0.0, 0.0)

    # TODO 5  Drive for 3 s: read the rear left motor and fill your column.
    for _ in range(30):
        # fb = car.LatestMotorFeedback(ci.Motor.RearLeft)
        # if fb is not None:
        #     print(fb.speed_rpm, fb.current_A, fb.temperature_C)
        #     car.AddLogValue(CURRENT_COLUMN, fb.current_A)
        time.sleep(0.1)

    # TODO 6  Stop safely: motors to 0, wheels straight, lights off.

    car.DisableLogging()
    car.Shutdown()

    # TODO 7  Check your work - uncomment once TODO 2 logs something.
    # check_log()


if __name__ == "__main__":
    main()
