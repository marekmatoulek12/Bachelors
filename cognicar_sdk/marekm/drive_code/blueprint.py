"""Follow the reference path. Run from the SDK root: python3 exercises/autopilot.py [--sim]"""
import argparse
import signal
import time

import cognicar_interface as ci
import cogni_tracking as tt
from marekm.imports import PoseVelKalman
from student_low_level_control import StudentLowLevelControl

PERIOD_S = 0.01

class ConnectionBlueprint():
    def __init__(self, sim: bool = False) -> None:
        self.kf = PoseVelKalman()
        self.controller = StudentLowLevelControl()
        cfg = ci.Config()
        cfg.backend = ci.Backend.Ros
        cfg.simulation = sim
        car = ci.Cognicar()

        if not car.Initialize(cfg):
            raise SystemExit("Could not connect to the car or simulator.")
        if not car.SetLidarEnabled(True):
            raise SystemExit("Could not start LiDAR localization.")
        
        print("Waiting for feedback.", flush=True)
        self.localized = False
        self.last_fix = 0
        self.driving = False
        self.next_tick = time.monotonic()
    

    def run(self):

        try:
            


            while True:
                next_tick += PERIOD_S
                # The pose fix: LiDAR localization on the car, the simulator's own pose in --sim.
                fix = car.LatestModel() if args.sim else car.LatestLocalization(max_age_ms=1000)
                if fix is not None and fix.sample.sequence != last_fix:
                    last_fix = fix.sample.sequence
                    o = fix.pose.orientation
                    fix_yaw = tt.yaw_rad(o.x, o.y, o.z, o.w)
                    if localized:
                        kf.update_gps_pose(fix.pose.position.x, fix.pose.position.y, fix_yaw)
                    else:
                        kf.reset(fix.pose.position.x, fix.pose.position.y, fix_yaw)
                        localized = True

                imu = car.LatestImu()
                left = car.LatestWheelEncoder(ci.Motor.FrontLeft)
                right = car.LatestWheelEncoder(ci.Motor.FrontRight)
                servo = car.LatestServoFeedback()
                ready = localized and all(value is not None for value in (fix, imu, left, right))
                if ready != driving:
                    print("Driving." if ready else "Feedback lost. Car stopped, waiting.", flush=True)
                    driving = ready
                if ready:
                    kf.predict(PERIOD_S, imu.linear_acceleration_ms2.x, imu.linear_acceleration_ms2.y, imu.angular_velocity_rads.z)
                    kf.update_front_wheel_encoders(left.speed_rpm, right.speed_rpm, servo.angle_rad, imu.angular_velocity_rad.z)
                    state = kf.state # [x, y, yaw, vx, vy]

                    steering, rl_rpm, rr_rpm = test_loop(state, test_state)

                    
                    if not car.SetSteeringAngleRad(steering) or not car.SetRearSpeedRpm(rl_rpm, rr_rpm):
                        raise SystemExit("Could not send the drive command.")
                else:
                    car.SetRearCurrentA(0.0, 0.0)
                    car.SetSteeringAngleRad(0.0)
                    localized = False
                    controller.reset()
                time.sleep(max(0.0, next_tick - time.monotonic()))
        except KeyboardInterrupt:
            pass
        finally:
            car.Shutdown()
            print("Stopped.", flush=True)


## THIS IS HOW YOU RUN A TEST AFTERWARDS
def request_stop(*_):
    raise KeyboardInterrupt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true", help="drive the simulator instead of the car")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)

    test_run = Test(sim = args.sim)

    main()
