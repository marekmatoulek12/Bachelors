"""Follow the reference path. Run from the SDK root: python3 exercises/autopilot.py [--sim]"""
import argparse
import signal
import time

import cognicar_interface as ci
import cogni_tracking as tt
from student_kalman import StudentKalman
from student_tracker import StudentTracker
from student_low_level_control import StudentLowLevelControl

PATH_FILE = "data/oval_path_clockwise.npy"
TARGET_SPEED_MPS = 1.0
PERIOD_S = 0.01


def request_stop(*_):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true", help="drive the simulator instead of the car")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)

    path = tt.load_path(PATH_FILE)
    kalman = StudentKalman()
    tracker = StudentTracker()
    controller = StudentLowLevelControl()
    cfg = ci.Config()
    cfg.backend = ci.Backend.Ros
    cfg.simulation = args.sim
    car = ci.Cognicar()
    if not car.Initialize(cfg):
        raise SystemExit("Could not connect to the car or simulator.")
    try:
        if not car.SetLidarEnabled(True):
            raise SystemExit("Could not start LiDAR localization.")
        print("Waiting for feedback.", flush=True)
        localized = False
        last_fix = 0
        driving = False
        next_tick = time.monotonic()
        while True:
            next_tick += PERIOD_S
            # The pose fix: LiDAR localization on the car, the simulator's own pose in --sim.
            fix = car.LatestModel() if args.sim else car.LatestLocalization(max_age_ms=1000)
            if fix is not None and fix.sample.sequence != last_fix:
                last_fix = fix.sample.sequence
                o = fix.pose.orientation
                fix_yaw = tt.yaw_rad(o.x, o.y, o.z, o.w)
                if localized:
                    kalman.update(fix.pose.position.x, fix.pose.position.y, fix_yaw)
                else:
                    kalman.reset(fix.pose.position.x, fix.pose.position.y, fix_yaw)
                    localized = True

            imu = car.LatestImu()
            left = car.LatestWheelEncoder(ci.Motor.FrontLeft)
            right = car.LatestWheelEncoder(ci.Motor.FrontRight)
            ready = localized and all(value is not None for value in (fix, imu, left, right))
            if ready != driving:
                print("Driving." if ready else "Feedback lost. Car stopped, waiting.", flush=True)
                driving = ready
            if ready:
                kalman.predict(left.speed_rpm, right.speed_rpm, imu.angular_velocity_rads.z)
                pose = tt.make_planar_pose(kalman.x, kalman.y, kalman.yaw)
                twist = tt.make_planar_twist(kalman.speed)
                steering = tracker.step(path, pose, twist)
                currents = controller.step(steering, TARGET_SPEED_MPS, left.speed_rpm, right.speed_rpm,
                                           imu.angular_velocity_rads.z, PERIOD_S)
                if not car.SetSteeringAngleRad(steering) or not car.SetRearCurrentA(*currents):
                    raise SystemExit("Could not send the drive command.")
            else:
                car.SetRearCurrentA(0.0, 0.0)
                car.SetSteeringAngleRad(0.0)
                localized = False
                tracker.reset()
                controller.reset()
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        car.Shutdown()
        print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
