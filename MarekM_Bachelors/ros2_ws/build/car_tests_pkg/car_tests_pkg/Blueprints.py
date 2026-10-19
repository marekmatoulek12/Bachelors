import csv
import time
import cognicar_interface as ci
import rclpy
from rclpy.duration import Duration
from pathlib import Path
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from typing import Callable
import numpy as np
from math import atan2, isfinite, pi, cos, sin
from .Class_Blueprints import PIDGains, TwinTrackParams

# source /home/cogni/cognicar_sdk/env.sh
# source install/setup.bash

import cognicar_interface as ci



class TestBluePrint(Node):

    ## --- CORE ---
    def __init__(self, node_name: str, 
                 control_loop: Callable[[], bool],
                 timer_period: float,
                 use_fw_encoders: bool = True
                 ):
        super().__init__(node_name)

        cogni_cfg = ci.Config()
        cogni_cfg.backend = ci.Backend.Ros
        cogni_cfg.simulation = False
        self.cogni_car = ci.Cognicar()
        if not self.cogni_car.Initialize(cogni_cfg):
            raise SystemExit("Could not connect to the car or simulator.")
        if not self.cogni_car.SetLidarEnabled(True):
            raise SystemExit("Could not start LiDAR localization.")

        self.car = TwinTrackParams()
        self.control_loop = control_loop
        self.should_stop = False
        self.ekf = VelocityEKF()

        self.use_fw_encoders = use_fw_encoders

        # Proměnné pro sledování času v callbackcích
        self.last_imu_time = None
        self.last_gps_time = None

        # Proměnné pro derivaci GPS
        self.last_gps_x = None
        self.last_gps_y = None

        workspace_root = Path(__file__).resolve().parents[3]
        logs_dir = workspace_root / 'CSV_logs'
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.timer = self.create_timer(timer_period, self._timer_callback)

        # Publishers pro rizeni
        self.pub_steer = self.create_publisher(Float64, '/servo/front/angle_req', 10)
        self.pub_rpm_rl = self.create_publisher(Float64, '/powertrain/rl/rpm_req', 10)
        self.pub_rpm_rr = self.create_publisher(Float64, '/powertrain/rr/rpm_req', 10)

        # Otazka, jestli brat subscription z state/odometry nebo z gps/pose a map/twist
        # odometry poskytuje vsechno najednou -> automaticky casove synchronizovane + je to modernejsi pristup
        # issue je ze podle logu to vypada, ze z auta jde jen gps/pose a map/twist

        # Subscribers pro zpetnou vazbu               
        self.sub_gps = self.create_subscription(
            PoseStamped,
            '/gps/pose',
            self.gps_callback,
            10
        )
        self.sub_imu = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        self.sub_rl_rpm = self.create_subscription(
            Float64,
            '/powertrain/rl/rpm_fb',
            self.rl_callback,
            10
        )
        self.sub_rr_rpm = self.create_subscription(
            Float64,
            '/powertrain/rr/rpm_fb',
            self.rr_callback,
            10
        )
        self.sub_fl_rpm = self.create_subscription(
            Float64,
            '/powertrain/fl/wheel_enc', ## idk jestli topic dobre
            self.fl_callback,
            10
        )
        self.sub_fr_rpm = self.create_subscription(
            Float64,
            '/powertrain/fr/wheel_enc', ## idk jestli topic dobre
            self.fr_callback,
            10
        )
        
        self.sub_steer_rad_fb = self.create_subscription(
            Float64,
            '/servo/front/angle_fb',
            self.servo_callback,
            10
        )

        # Nastaveni logovani do CSV
        self.log_filename = str(logs_dir / f'{self.get_name()}_log_{int(time.time())}.csv')
        self.log_file = open(self.log_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        # Hlavicka CSV
        self.csv_writer.writerow([
            'timestamp',
            'cmd_rpm',
            'cmd_steer_rad',
            'fb_steer_rad',
            'gps_x',
            'gps_y', 
            'gps_yaw',
            'vx',
            'vy',
            'yaw_rate',
            'accel_x',
            'accel_y',
            'rl_rpm',
            'rr_rpm',
            'fl_rpm',
            'fr_rpm'
            ])
        
        self.get_logger().info(f"Test '{self.get_name()}' inicializovan.")

        self.gps_x          = None
        self.gps_y          = None
        self.gps_yaw        = None
        self.vx             = None
        self.vy             = None
        self.yaw_rate       = None
        self.acc_x          = None
        self.acc_y          = None
        self.rl_rpm_fb      = None
        self.rr_rpm_fb      = None
        self.fl_rpm         = None
        self.fr_rpm         = None
        self.cmd_steer_rad  = None
        self.fb_steer_rad   = None
        self._missing_data_warning = False

        self.p_fl = (self.car.front_axle_len, self.car.front_track_width / 2)
        self.p_fr = (self.car.front_axle_len, -self.car.front_track_width / 2)
        self.p_rl = (self.car.rear_axle_len, self.car.rear_track_width / 2)
        self.p_rr = (self.car.rear_axle_len, -self.car.rear_track_width / 2)

        self.START_TIME = self.get_clock().now().nanoseconds / 1e9

    def _timer_callback(self):
        keep_running = self.control_loop()
        if not keep_running:
            self.should_stop = True

    ## --- CALLBACKS ---
    # velocity ekf update - GPS
    def gps_callback(self, msg):
        # Aktualni pozice auta z GPS
        if msg is None:
            self.gps_x = float("nan")
            self.gps_y = float("nan")
            self.gps_yaw = float("nan")
            return
    
        current_time = self.get_clock().now().nanoseconds / 1e9
        current_x = msg.pose.position.x
        current_y = msg.pose.position.y
        self.gps_yaw = self.get_yaw_from_quaternion(msg.pose.orientation)
        
        if self.last_gps_time is not None and self.last_gps_x is not None:
            dt_gps = current_time - self.last_gps_time
            if dt_gps > 0:
                # 1. Rychlost v globálním souřadném systému mapy
                v_X_global = (current_x - self.last_gps_x) / dt_gps
                v_Y_global = (current_y - self.last_gps_y) / dt_gps
                
                # 2. Transformační matice (otočení o -yaw do soustavy auta)
                c, s = cos(-self.gps_yaw), sin(-self.gps_yaw)
                z_vx_gps = c * v_X_global - s * v_Y_global
                z_vy_gps = s * v_X_global + c * v_Y_global
                
                # 3. EKF Update z GPS (s fixní maticí šumu GPS)
                self.ekf.update(z_vx_gps, z_vy_gps, self.ekf.R_gps)

        # Uložení pro příští krok
        self.last_gps_x = current_x
        self.last_gps_y = current_y
        self.last_gps_time = current_time
        
        self.gps_x = current_x
        self.gps_y = current_y

    # velocity ekf prediction - IMU
    def imu_callback(self, msg):
        if msg is None:
            self.acc_x = float("nan")
            self.acc_y = float("nan")
            self.yaw_rate = float("nan")
            return

        # maji spatne souradny system
        #        x                y
        #        ^                ^  
        #        |                |  
        # y <----|----   <--- ----|----> x
        #        |                |
        #        |                |

        self.acc_x = msg.linear_acceleration.y
        self.acc_y = -msg.linear_acceleration.x
        self.yaw_rate = msg.angular_velocity.z

        # EKF Predikce
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_imu_time is not None:
            dt = current_time - self.last_imu_time
            self.ekf.predict(dt, self.acc_x, self.acc_y, self.yaw_rate)
            
            # Aktualizace globálních proměnných pro logování a řízení
            self.vx = self.ekf.x[0]
            self.vy = self.ekf.x[1]
            
        self.last_imu_time = current_time

    # velocity ekf update - front wheel encoders
    def fl_callback(self, msg):
        if msg is None:
            self.fl_rpm = float("nan")
            return


        self.fl_rpm = msg.data ## je to RPM nebo rad/s nebo?
        self.update_ekf_from_wheels(use= self.use_fw_encoders)

    def fr_callback(self, msg):
        if msg is None:
            self.fr_rpm = float("nan")
            return

        self.fr_rpm = msg.data
        self.update_ekf_from_wheels(use= self.use_fw_encoders)

    def rl_callback(self, msg):
        # Aktualni zmerena rpm kol
        if msg is None:
            self.rl_rpm_fb = float("nan")
            return

        self.rl_rpm_fb = msg.data / 2

    def rr_callback(self, msg):
        # Aktualni zmerena rpm kol
        if msg is None:
            self.rr_rpm_fb = float("nan")
            return

        self.rr_rpm_fb = -msg.data / 2 ## spatne nastavene motory

    def servo_callback(self, msg):
        if msg is None:
            self.fb_steer_rad = float("nan")
            return
        self.fb_steer_rad = msg.data
        
    ## --- DATA HANDLING ---
    def log_commands(self, cmd_rpm, cmd_steer_rad):
        self.cmd_steer_rad = cmd_steer_rad 
        self.cmd_rpm = cmd_rpm

    def log_data(self, cmd_rpm, cmd_steer_rad):
        # Logovani dat do CSV
        current_time = self.get_clock().now().nanoseconds / 1e9
        ellapsed_time = current_time - self.START_TIME
        self.csv_writer.writerow([
            ellapsed_time,
            cmd_rpm,
            cmd_steer_rad,
            self.fb_steer_rad,
            self.gps_x,
            self.gps_y,
            self.gps_yaw,
            self.vx,
            self.vy,
            self.yaw_rate,
            self.acc_x,
            self.acc_y,
            self.rl_rpm_fb,
            self.rr_rpm_fb,
            self.fl_rpm,
            self.fr_rpm
        ])

    def publish_data(self, cmd_rpm, cmd_steer_rad):
        # Publikovani dat do ROS topicu
        msg_steer = Float64()
        msg_steer.data = cmd_steer_rad
        self.pub_steer.publish(msg_steer)

        msg_rpm_rl = Float64()
        msg_rpm_rr = Float64()
        # prevodovka - cmd_rpm na zadni kola = 2 * rpm na motory
        msg_rpm_rl.data = 2 * cmd_rpm
        msg_rpm_rr.data = -2 * cmd_rpm # motory jsou rizeny opacne
        self.pub_rpm_rl.publish(msg_rpm_rl)
        self.pub_rpm_rr.publish(msg_rpm_rr)

    def destroy_blueprint_node(self):
        # Zastaveni timeru a ukonceni node
        if self.timer is not None:
            self.timer.cancel()
        self.should_stop = True

        # Pri Ctrl+C muze byt ROS kontext uz neplatny, proto nepublikujeme, pokud nefunguje.
        try:
            if rclpy.ok():
                # snaha zastavit auto
                self.publish_data(0.0, 0.0)
                self.get_logger().info("Resetovaci zprava odeslana")

                timeout = Duration(seconds=1.0)
                for publisher in (self.pub_steer, self.pub_rpm_rl, self.pub_rpm_rr):
                    if not publisher.wait_for_all_acked(timeout):
                        self.get_logger().warning(
                            f"Nepřišlo potvrzení doručení z topicu {publisher.topic_name}."
                        )
                        
        except Exception:
            self.get_logger().warning("Shutdown publish selhal, pokracuji v ukonceni node.")
        finally:
            try:
                self.cogni_car.Shutdown()
                if hasattr(self, 'log_file') and self.log_file is not None:
                    self.log_file.close()
            finally:
                self.get_logger().info(f"Test ukoncen. Data ulozena do {self.log_filename}")
                super().destroy_node()

    def all_data_available(self):
        values = (
            self.gps_x,
            self.gps_y,
            self.gps_yaw,
            self.vx,
            self.vy,
            self.yaw_rate,
            self.acc_x,
            self.acc_y,
            self.rl_rpm_fb,
            self.rr_rpm_fb,
            self.fl_rpm,
            self.fr_rpm,
            # self.fb_steer_rad
            )
        if any(v is None or not isfinite(float(v)) for v in values):
            if not self._missing_data_warning:
                self.get_logger().warning("Nektere data nejsou k dispozici nebo nejsou platna.")
                self._missing_data_warning = True
            return False
        self._missing_data_warning = False
        return True

    ## --- HELPER FUNCTIONS ---
    def update_slips(self):
        if self.all_data_available():
            rl_rad_s = self.rpm2rad_s(self.rl_rpm_fb) # type: ignore
            rr_rad_s = self.rpm2rad_s(self.rr_rpm_fb) # type: ignore

            vx_rl = self.vx - self.yaw_rate * self.p_rl[1] # type: ignore
            vy_rl = self.vy + self.yaw_rate * self.p_rl[0] # type: ignore
            vx_rr = self.vx - self.yaw_rate * self.p_rr[1] # type: ignore
            vy_rr = self.vx + self.yaw_rate * self.p_rr[0] # type: ignore

            v_ref_rl = max(abs(vx_rl), self.car.rear_wheel_radius * rl_rad_s, 0.2) # type: ignore
            v_ref_rr = max(abs(vx_rr), self.car.rear_wheel_radius * rr_rad_s, 0.2) # type: ignore

            alpha_rl = atan2(vy_rl, v_ref_rl)
            k_rl = (self.car.rear_wheel_radius * rl_rad_s - vx_rl) / v_ref_rl # type: ignore
            alpha_rr = atan2(vy_rr, v_ref_rr)
            k_rr = (self.car.rear_wheel_radius * rr_rad_s - vx_rr) / v_ref_rr # type: ignore

            return alpha_rl, k_rl, alpha_rr, k_rr
        return None, None, None, None

    def update_ekf_from_wheels(self, use: bool = True):
        # Máme všechna potřebná data?
        if (self.fl_rpm is not None and self.fr_rpm is not None and 
            self.cmd_steer_rad is not None and self.yaw_rate is not None):
            
            speed = self.rpm2vx((self.fl_rpm + self.fr_rpm) / 2)
            steering_angle = self.cmd_steer_rad
            
            # Kinematický výpočet
            z_vx = cos(steering_angle) * speed
            z_vy = sin(steering_angle) * speed - (self.yaw_rate * self.car.front_axle_len)
            
            # Heuristika pro detekci driftu - akcelerace Y nad 4 m/s^2 nebo obrovský yaw_rate
            is_slipping = (self.acc_y is not None and abs(self.acc_y) > 4.0) or \
                        (self.yaw_rate is not None and abs(self.yaw_rate) > 2.0)
            
            # Dynamické určení matice R
            if is_slipping or not use:
                # Obrovský šum -> EKF bude kola ignorovat a pojede jen na IMU + GPS
                current_R = self.ekf.R_wheels * 1000.0 
            else:
                current_R = self.ekf.R_wheels
                
            self.ekf.update(z_vx, z_vy, current_R)

    def vx2rpm(self, vx: float):
        return vx * 60 / (2 * pi * self.car.rear_wheel_radius)

    def rpm2vx(self, rpm: float):
        return rpm * (2 * pi * self.car.rear_wheel_radius) / 60

    @staticmethod
    def deg2rad(angle: float):
        return angle * pi / 180

    @staticmethod
    def rad2deg(angle: float):
        return angle * 180 / pi

    @staticmethod
    def get_yaw_from_quaternion(q):
        """Převod ROS 2 geometry_msgs/msg/Quaternion na úhel yaw v radiánech."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def rpm2rad_s(rpm: float | np.ndarray) -> float | np.ndarray:
        return rpm * (2.0 * np.pi / 60.0)

class PIDController:
    """Discrete PID controller with clamping and simple anti-windup."""

    def __init__(
        self,
        gains: PIDGains,
        output_limit: tuple[float, float] | None = None,
        integrator_limit: tuple[float, float] | None = None,
    ) -> None:
        self.gains = gains
        self.output_limit = output_limit
        self.integrator_limit = integrator_limit
        self.integral = 0.0
        self.previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def step(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        derivative = 0.0
        if self.previous_error is not None and self.gains.kd != 0.0:
            derivative = (error - self.previous_error) / dt

        candidate_integral = self.integral + error * dt
        if self.integrator_limit is not None:
            candidate_integral = self.clamp(candidate_integral, self.integrator_limit[0], self.integrator_limit[1])

        unclamped = (
            self.gains.kp * error
            + self.gains.ki * candidate_integral
            + self.gains.kd * derivative
        )
        output = unclamped
        if self.output_limit is not None:
            output = self.clamp(output, self.output_limit[0], self.output_limit[1])

        # Accept the new integral when unsaturated, or when it helps move the
        # saturated output back toward the allowed range.
        if output == unclamped or (unclamped > output and error < 0.0) or (unclamped < output and error > 0.0):
            self.integral = candidate_integral

        self.previous_error = error
        return output

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

class VelocityEKF:
    def __init__(self):
        # Stavy: [vx, vy]
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1.0
        
        self.Q = np.eye(2) * 0.05       # Šum modelu (IMU)
        self.R_wheels = np.diag([0.1, 10]) # Šum kol (když nekloužou)
        self.R_gps = np.eye(2) * 20    # Šum GPS (vyhlazení)

    def predict(self, dt: float, ax: float, ay: float, yaw_rate: float):
        if dt <= 0: return
        
        # Fyzikální model pro rychlosti těžiště
        vx_new = self.x[0] + (ax + yaw_rate * self.x[1]) * dt
        vy_new = self.x[1] + (ay - yaw_rate * self.x[0]) * dt
        
        # Jakobián (změna vůči dt)
        F = np.array([
            [1.0,  yaw_rate * dt],
            [-yaw_rate * dt, 1.0]
        ])
        
        self.x = np.array([vx_new, vy_new])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_vx: float, z_vy: float, R_matrix: np.ndarray):
        z = np.array([z_vx, z_vy])
        H = np.eye(2) # Měříme přímo vx, vy
        
        # Inovace a Kalmanův zisk
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R_matrix
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
