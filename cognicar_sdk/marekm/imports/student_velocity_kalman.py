"""Two-dimensional velocity EKF fusing IMU, GPS, and front wheel encoders.

The state is the vehicle-frame velocity ``[vx, vy]`` in m/s. Call ``predict``
with each IMU sample, ``update_gps_position`` with planar GPS poses and their
timestamps, and ``update_front_wheel_encoders`` with encoder RPM and steering.
"""

from __future__ import annotations

import math

Matrix2 = list[list[float]]


class StudentVelocityKalman:
    """Estimate vehicle-frame ``vx`` and ``vy`` with a two-state EKF."""

    def __init__(
        self,
        wheel_radius_m: float = 0.033,
        front_axle_distance_m: float = 0.16,
        initial_variance: float = 1.0,
        process_noise_density: float = 0.05,
        gps_velocity_variance: float = 0.1,
        wheel_velocity_variance: float = 0.1,
        slipping_noise_multiplier: float = 1000.0,
    ) -> None:
        self.wheel_radius_m = self._positive(wheel_radius_m, "wheel_radius_m")
        self.front_axle_distance_m = self._nonnegative(
            front_axle_distance_m, "front_axle_distance_m"
        )
        self.initial_variance = self._positive(initial_variance, "initial_variance")
        process_noise_density = self._nonnegative(
            process_noise_density, "process_noise_density"
        )
        gps_velocity_variance = self._positive(
            gps_velocity_variance, "gps_velocity_variance"
        )
        wheel_velocity_variance = self._positive(
            wheel_velocity_variance, "wheel_velocity_variance"
        )
        self.slipping_noise_multiplier = self._positive(
            slipping_noise_multiplier, "slipping_noise_multiplier"
        )

        self.x = [0.0, 0.0]
        self.P = self._diagonal(self.initial_variance)
        self.Q = self._diagonal(process_noise_density)
        self.R_gps = self._diagonal(gps_velocity_variance)
        self.R_wheels = self._diagonal(wheel_velocity_variance)
        self._last_gps: tuple[float, float, float] | None = None
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
    def _diagonal(value: float) -> Matrix2:
        return [[value, 0.0], [0.0, value]]

    @staticmethod
    def _transpose(matrix: Matrix2) -> Matrix2:
        return [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]

    @staticmethod
    def _multiply(left: Matrix2, right: Matrix2) -> Matrix2:
        return [
            [
                left[row][0] * right[0][column]
                + left[row][1] * right[1][column]
                for column in range(2)
            ]
            for row in range(2)
        ]

    @staticmethod
    def _add(left: Matrix2, right: Matrix2, right_scale: float = 1.0) -> Matrix2:
        return [
            [
                left[row][column] + right_scale * right[row][column]
                for column in range(2)
            ]
            for row in range(2)
        ]

    @staticmethod
    def _scale(matrix: Matrix2, scale: float) -> Matrix2:
        return [
            [matrix[row][column] * scale for column in range(2)]
            for row in range(2)
        ]

    @property
    def vx(self) -> float:
        return self.x[0]

    @property
    def vy(self) -> float:
        return self.x[1]

    @property
    def velocity(self) -> tuple[float, float]:
        """Return the current ``(vx, vy)`` estimate."""
        return self.vx, self.vy

    def reset(self, vx: float = 0.0, vy: float = 0.0) -> None:
        """Reset the velocity estimate and discard the GPS differencing sample."""
        self.x = [self._finite(vx, "vx"), self._finite(vy, "vy")]
        self.P = self._diagonal(self.initial_variance)
        self._last_gps = None
        self._last_accel_y = None

    def predict(
        self,
        dt_s: float,
        accel_x_mps2: float,
        accel_y_mps2: float,
        yaw_rate_radps: float,
    ) -> tuple[float, float]:
        """Propagate velocity using body-frame IMU acceleration and yaw rate."""
        dt_s = self._positive(dt_s, "dt_s")
        ax = self._finite(accel_x_mps2, "accel_x_mps2")
        ay = self._finite(accel_y_mps2, "accel_y_mps2")
        yaw_rate = self._finite(yaw_rate_radps, "yaw_rate_radps")
        self._last_accel_y = ay

        vx, vy = self.x
        self.x = [
            vx + (ax + yaw_rate * vy) * dt_s,
            vy + (ay - yaw_rate * vx) * dt_s,
        ]
        F = [[1.0, yaw_rate * dt_s], [-yaw_rate * dt_s, 1.0]]
        self.P = self._add(
            self._multiply(self._multiply(F, self.P), self._transpose(F)),
            self.Q,
            dt_s,
        )
        self._symmetrize_covariance()
        return self.velocity

    def update_gps_position(
        self, x_m: float, y_m: float, yaw_rad: float, timestamp_s: float
    ) -> bool:
        """Update from a planar GPS pose; return False for the first GPS sample.

        Consecutive positions are differenced into a global velocity and rotated
        into the vehicle frame using the current pose yaw.
        """
        x_m = self._finite(x_m, "x_m")
        y_m = self._finite(y_m, "y_m")
        yaw_rad = self._finite(yaw_rad, "yaw_rad")
        timestamp_s = self._finite(timestamp_s, "timestamp_s")

        if self._last_gps is None:
            self._last_gps = (x_m, y_m, timestamp_s)
            return False

        last_x, last_y, last_timestamp = self._last_gps
        dt_s = timestamp_s - last_timestamp
        if dt_s <= 0.0:
            raise ValueError("GPS timestamps must increase strictly")

        global_vx = (x_m - last_x) / dt_s
        global_vy = (y_m - last_y) / dt_s
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        body_vx = c * global_vx + s * global_vy
        body_vy = -s * global_vx + c * global_vy
        self._update((body_vx, body_vy), self.R_gps)
        self._last_gps = (x_m, y_m, timestamp_s)
        return True

    def update_front_wheel_encoders(
        self,
        front_left_rpm: float,
        front_right_rpm: float,
        steering_angle_rad: float,
        yaw_rate_radps: float,
        slipping: bool | None = None,
    ) -> tuple[float, float]:
        """Update from front-wheel RPM, steering angle, and current yaw rate.

        The encoder-derived speed and steering angle form a vehicle-frame
        velocity proxy, with the front axle's lateral yaw velocity removed.
        By default, increase wheel measurement noise for lateral acceleration
        above 4 m/s² or yaw rate above 2 rad/s. ``slipping`` overrides this
        heuristic when set to True or False.
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

        average_rpm = 0.5 * (left_rpm + right_rpm)
        speed_mps = average_rpm * (2.0 * math.pi / 60.0) * self.wheel_radius_m
        measurement = (
            math.cos(steering) * speed_mps,
            math.sin(steering) * speed_mps
            - yaw_rate * self.front_axle_distance_m,
        )
        R = self.R_wheels
        if slipping:
            R = self._scale(R, self.slipping_noise_multiplier)
        self._update(measurement, R)
        return self.velocity

    def _update(self, measurement: tuple[float, float], R: Matrix2) -> None:
        innovation = [measurement[0] - self.x[0], measurement[1] - self.x[1]]
        S = self._add(self.P, R)
        determinant = S[0][0] * S[1][1] - S[0][1] * S[1][0]
        if not math.isfinite(determinant) or determinant <= 0.0:
            raise ArithmeticError("Velocity covariance is not invertible")

        inverse_S = [
            [S[1][1] / determinant, -S[0][1] / determinant],
            [-S[1][0] / determinant, S[0][0] / determinant],
        ]
        K = self._multiply(self.P, inverse_S)
        self.x = [
            self.x[row] + K[row][0] * innovation[0] + K[row][1] * innovation[1]
            for row in range(2)
        ]

        I_minus_K = [
            [1.0 - K[0][0], -K[0][1]],
            [-K[1][0], 1.0 - K[1][1]],
        ]
        self.P = self._add(
            self._multiply(
                self._multiply(I_minus_K, self.P), self._transpose(I_minus_K)
            ),
            self._multiply(self._multiply(K, R), self._transpose(K)),
        )
        self._symmetrize_covariance()

    def _symmetrize_covariance(self) -> None:
        average = 0.5 * (self.P[0][1] + self.P[1][0])
        self.P[0][1] = average
        self.P[1][0] = average
