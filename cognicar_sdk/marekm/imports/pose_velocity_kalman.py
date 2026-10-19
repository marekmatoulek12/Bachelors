"""Pose EKF fusing IMU motion, front-wheel encoders, and GPS pose fixes.

The state is ``[x, y, yaw, vx, vy]``: pose in the map frame and velocity in the
vehicle frame. IMU samples propagate the state; GPS poses and front-wheel
measurements correct it.
"""

from __future__ import annotations

import math

Vector5 = list[float]
Matrix5 = list[list[float]]


class PoseVelKalman:
    """Estimate filtered planar pose and vehicle-frame velocity."""

    def __init__(
        self,
        wheel_radius_m: float = 0.033,
        front_axle_distance_m: float = 0.16,
        front_track_width_m: float = 0.16,
        initial_pose_variance: float = 1.0,
        initial_velocity_variance: float = 1.0,
        position_process_noise_density: float = 2e-5,
        yaw_process_noise_density: float = 1e-5,
        velocity_process_noise_density: float = 0.05,
        gps_position_variance: float = 2.5e-3,
        gps_yaw_variance: float = 2.5e-3,
        wheel_speed_variance: float = 0.1,
        slipping_noise_multiplier: float = 1000.0,
        use_imu_acceleration: bool = True,
    ) -> None:
        self.wheel_radius_m = self._positive(wheel_radius_m, "wheel_radius_m")
        self.front_axle_distance_m = self._nonnegative(
            front_axle_distance_m, "front_axle_distance_m"
        )
        self.front_track_width_m = self._nonnegative(
            front_track_width_m, "front_track_width_m"
        )
        self.initial_pose_variance = self._positive(
            initial_pose_variance, "initial_pose_variance"
        )
        self.initial_velocity_variance = self._positive(
            initial_velocity_variance, "initial_velocity_variance"
        )
        self.position_process_noise_density = self._nonnegative(
            position_process_noise_density, "position_process_noise_density"
        )
        self.yaw_process_noise_density = self._nonnegative(
            yaw_process_noise_density, "yaw_process_noise_density"
        )
        self.velocity_process_noise_density = self._nonnegative(
            velocity_process_noise_density, "velocity_process_noise_density"
        )
        self.gps_position_variance = self._positive(
            gps_position_variance, "gps_position_variance"
        )
        self.gps_yaw_variance = self._positive(
            gps_yaw_variance, "gps_yaw_variance"
        )
        self.wheel_speed_variance = self._positive(
            wheel_speed_variance, "wheel_speed_variance"
        )
        self.slipping_noise_multiplier = self._positive(
            slipping_noise_multiplier, "slipping_noise_multiplier"
        )
        if not isinstance(use_imu_acceleration, bool):
            raise TypeError("use_imu_acceleration must be a bool")
        self.use_imu_acceleration = use_imu_acceleration

        self.state: Vector5 = [0.0] * 5
        self.P = self._diagonal(
            [
                self.initial_pose_variance,
                self.initial_pose_variance,
                self.initial_pose_variance,
                self.initial_velocity_variance,
                self.initial_velocity_variance,
            ]
        )
        self._last_accel_y: float | None = None

    @staticmethod
    def _finite(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        result = cls._finite(value, name)
        if result <= 0.0:
            raise ValueError(f"{name} must be greater than zero")
        return result

    @classmethod
    def _nonnegative(cls, value: float, name: str) -> float:
        result = cls._finite(value, name)
        if result < 0.0:
            raise ValueError(f"{name} must not be negative")
        return result

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _diagonal(values: list[float]) -> Matrix5:
        size = len(values)
        return [
            [values[row] if row == column else 0.0 for column in range(size)]
            for row in range(size)
        ]

    @staticmethod
    def _transpose(matrix: Matrix5) -> Matrix5:
        return [list(column) for column in zip(*matrix)]

    @staticmethod
    def _multiply(left: Matrix5, right: Matrix5) -> Matrix5:
        rows = len(left)
        inner = len(right)
        columns = len(right[0])
        return [
            [
                sum(left[row][k] * right[k][column] for k in range(inner))
                for column in range(columns)
            ]
            for row in range(rows)
        ]

    @property
    def x(self) -> float:
        return self.state[0]

    @property
    def y(self) -> float:
        return self.state[1]

    @property
    def yaw(self) -> float:
        return self.state[2]

    @property
    def vx(self) -> float:
        return self.state[3]

    @property
    def vy(self) -> float:
        return self.state[4]

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.yaw

    @property
    def velocity(self) -> tuple[float, float]:
        return self.vx, self.vy

    def reset(
        self, x: float, y: float, yaw_rad: float, vx: float = 0.0, vy: float = 0.0
    ) -> None:
        """Initialize pose from a trusted fix and clear velocity history."""
        self.state = [
            self._finite(x, "x"),
            self._finite(y, "y"),
            self._wrap_angle(self._finite(yaw_rad, "yaw_rad")),
            self._finite(vx, "vx"),
            self._finite(vy, "vy"),
        ]
        self.P = self._diagonal(
            [
                self.initial_pose_variance,
                self.initial_pose_variance,
                self.initial_pose_variance,
                self.initial_velocity_variance,
                self.initial_velocity_variance,
            ]
        )
        self._last_accel_y = None

    def predict(
        self,
        dt_s: float,
        accel_x_mps2: float,
        accel_y_mps2: float,
        yaw_rate_radps: float,
    ) -> tuple[float, float, float]:
        """Propagate pose and velocity using a planar IMU sample.

        Accelerometer inputs are assumed to be gravity-compensated and aligned
        with the vehicle axes. Set ``use_imu_acceleration=False`` if the sensor
        data has not been calibrated or gravity-compensated.
        """
        dt_s = self._positive(dt_s, "dt_s")
        ax = self._finite(accel_x_mps2, "accel_x_mps2")
        ay = self._finite(accel_y_mps2, "accel_y_mps2")
        yaw_rate = self._finite(yaw_rate_radps, "yaw_rate_radps")
        self._last_accel_y = ay

        x, y, yaw, vx, vy = self.state
        c = math.cos(yaw)
        s = math.sin(yaw)
        use_accel = self.use_imu_acceleration
        effective_ax = ax if use_accel else 0.0
        effective_ay = ay if use_accel else 0.0

        self.state = [
            x + (vx * c - vy * s) * dt_s,
            y + (vx * s + vy * c) * dt_s,
            self._wrap_angle(yaw + yaw_rate * dt_s),
            vx + (effective_ax + yaw_rate * vy) * dt_s,
            vy + (effective_ay - yaw_rate * vx) * dt_s,
        ]

        F = [
            [1.0, 0.0, (-vx * s - vy * c) * dt_s, c * dt_s, -s * dt_s],
            [0.0, 1.0, (vx * c - vy * s) * dt_s, s * dt_s, c * dt_s],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, yaw_rate * dt_s],
            [0.0, 0.0, 0.0, -yaw_rate * dt_s, 1.0],
        ]
        Q = self._diagonal(
            [
                self.position_process_noise_density * dt_s,
                self.position_process_noise_density * dt_s,
                self.yaw_process_noise_density * dt_s,
                self.velocity_process_noise_density * dt_s,
                self.velocity_process_noise_density * dt_s,
            ]
        )
        self.P = self._multiply(
            self._multiply(F, self.P), self._transpose(F)
        )
        for row in range(5):
            for column in range(5):
                self.P[row][column] += Q[row][column]
        self._symmetrize_covariance()
        return self.pose

    def update_gps_pose(
        self, x_m: float, y_m: float, yaw_rad: float
    ) -> tuple[float, float, float]:
        """Correct pose with a GPS/localization pose measurement."""
        measured = (
            self._finite(x_m, "x_m"),
            self._finite(y_m, "y_m"),
            self._wrap_angle(self._finite(yaw_rad, "yaw_rad")),
        )
        for index, value, variance in (
            (0, measured[0], self.gps_position_variance),
            (1, measured[1], self.gps_position_variance),
            (2, measured[2], self.gps_yaw_variance),
        ):
            H = [0.0] * 5
            H[index] = 1.0
            residual = value - self.state[index]
            if index == 2:
                residual = self._wrap_angle(residual)
            self._scalar_update(H, residual, variance)
        self.state[2] = self._wrap_angle(self.state[2])
        return self.pose

    def update_front_wheel_encoders(
        self,
        front_left_rpm: float,
        front_right_rpm: float,
        steering_angle_rad: float,
        yaw_rate_radps: float,
        slipping: bool | None = None,
    ) -> tuple[float, float]:
        """Correct velocity using each front wheel's rolling speed and steering.

        Each measurement accounts for the velocity at that wheel's location,
        including the yaw-rate contribution from its front-axle offset.
        """
        left_rpm = self._finite(front_left_rpm, "front_left_rpm")
        right_rpm = self._finite(front_right_rpm, "front_right_rpm")
        steering = self._finite(steering_angle_rad, "steering_angle_rad")
        yaw_rate = self._finite(yaw_rate_radps, "yaw_rate_radps")
        if slipping is None:
            high_lateral_acceleration = (
                self._last_accel_y is not None and abs(self._last_accel_y) > 4.0
            )
            slipping = high_lateral_acceleration or abs(yaw_rate) > 2.0
        elif not isinstance(slipping, bool):
            raise TypeError("slipping must be a bool or None")

        c = math.cos(steering)
        s = math.sin(steering)
        H = [0.0, 0.0, 0.0, c, s]
        variance = self.wheel_speed_variance
        if slipping:
            variance *= self.slipping_noise_multiplier
        rpm_to_speed = (2.0 * math.pi / 60.0) * self.wheel_radius_m
        half_track = 0.5 * self.front_track_width_m
        for rpm, lateral_offset in (
            (left_rpm, half_track),
            (right_rpm, -half_track),
        ):
            measured_speed = rpm * rpm_to_speed
            yaw_velocity_along_wheel = yaw_rate * (
                s * self.front_axle_distance_m - c * lateral_offset
            )
            predicted_speed = (
                c * self.vx + s * self.vy + yaw_velocity_along_wheel
            )
            self._scalar_update(H, measured_speed - predicted_speed, variance)
        return self.velocity

    def _scalar_update(
        self, H: list[float], residual: float, measurement_variance: float
    ) -> None:
        PHt = [
            sum(self.P[row][column] * H[column] for column in range(5))
            for row in range(5)
        ]
        innovation_variance = (
            sum(H[index] * PHt[index] for index in range(5)) + measurement_variance
        )
        if not math.isfinite(innovation_variance) or innovation_variance <= 0.0:
            raise ArithmeticError("Pose covariance is not positive definite")

        gain = [value / innovation_variance for value in PHt]
        self.state = [
            self.state[index] + gain[index] * residual for index in range(5)
        ]

        I_minus_KH = [
            [
                (1.0 if row == column else 0.0) - gain[row] * H[column]
                for column in range(5)
            ]
            for row in range(5)
        ]
        joseph_left = self._multiply(
            self._multiply(I_minus_KH, self.P), self._transpose(I_minus_KH)
        )
        for row in range(5):
            for column in range(5):
                joseph_left[row][column] += (
                    gain[row] * measurement_variance * gain[column]
                )
        self.P = joseph_left
        self._symmetrize_covariance()

    def _symmetrize_covariance(self) -> None:
        for row in range(5):
            for column in range(row + 1, 5):
                average = 0.5 * (self.P[row][column] + self.P[column][row])
                self.P[row][column] = average
                self.P[column][row] = average
