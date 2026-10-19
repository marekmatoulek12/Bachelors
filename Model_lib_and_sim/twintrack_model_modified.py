"""Twin-track vehicle model with Pacejka tires and traction ellipse limiting.

The model is intended for a small vehicle.  Each wheel can receive its own
drive torque, brake torque, and steering angle.  Wheel steering angles are
clamped to +/- 40 degrees.

State vector:
    x, y, yaw, vx, vy, yaw_rate, omega_fl, omega_fr, omega_rl, omega_rr

Units are SI: m, kg, s, rad, N, Nm.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import atan, atan2, cos, isfinite, radians, sin


@dataclass(frozen=True)
class TwinTrackParams:
    front_wheel_radius: float = 0.033
    rear_wheel_radius: float = 0.033
    front_axle_len: float = 0.16
    rear_axle_len: float = 0.14
    front_cornering_stiff: float = 16.0
    rear_cornering_stiff: float = 16.0
    mass: float = 2.5
    moment_of_inertia: float = 0.027
    st_min_velocity: float = 0.2

    # Extra parameters needed by a twin-track model.
    front_track_width: float = 0.16
    rear_track_width: float = 0.16
    wheel_inertia: float = 0.001
    tire_friction_coeff: float = 0.9
    rolling_resistance: float = 0.03
    # By was calibrated from recorded driving using rolling open-loop
    # predictions. Bx, C/E, and the friction coefficient retain their original
    # values because the recording did not identify them reliably.
    pacejka_bx: float = 10.0
    pacejka_cx: float = 1.9
    pacejka_ex: float = 0.97
    pacejka_by: float = 2.15
    pacejka_cy: float = 1.3
    pacejka_ey: float = -1.6
    max_steer_deg: float = 40.0
    gravity: float = 9.81


@dataclass
class TwinTrackState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    omega_fl: float = 0.0
    omega_fr: float = 0.0
    omega_rl: float = 0.0
    omega_rr: float = 0.0

@dataclass
class TwinTrackStateDerivatives:
    x_dot: float = 0.0
    y_dot: float = 0.0
    yaw_dot: float = 0.0
    vx_dot: float = 0.0
    vy_dot: float = 0.0
    yaw_rate_dot: float = 0.0
    omega_fl_dot: float = 0.0
    omega_fr_dot: float = 0.0
    omega_rl_dot: float = 0.0
    omega_rr_dot: float = 0.0

@dataclass(frozen=True)
class WheelForces:
    fx: float
    fy: float
    normal_force: float
    slip_ratio: float
    slip_angle: float

@dataclass
class WheelAllParams:
    fx: float
    fy: float
    normal_force: float
    slip_ratio: float
    slip_angle: float
    omega: float
    drive_torque: float
    break_torque: float


@dataclass(frozen=True)
class TwinTrackInput:
    """Per-wheel model inputs.

    Drive torque and brake torque are in Nm.  Steering angles are in radians.
    Brake torque is treated as a positive magnitude opposing wheel rotation.
    """

    drive_torque_fl: float = 0.0
    drive_torque_fr: float = 0.0
    drive_torque_rl: float = 0.0
    drive_torque_rr: float = 0.0
    brake_torque_fl: float = 0.0
    brake_torque_fr: float = 0.0
    brake_torque_rl: float = 0.0
    brake_torque_rr: float = 0.0
    steering_angle_fl: float = 0.0
    steering_angle_fr: float = 0.0
    steering_angle_rl: float = 0.0
    steering_angle_rr: float = 0.0


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0


@dataclass(frozen=True)
class RearTorqueCommand:
    left_torque: float
    right_torque: float
    slip_ratio_ref: float
    omega_ref_left: float
    omega_ref_right: float


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
            candidate_integral = clamp(candidate_integral, self.integrator_limit[0], self.integrator_limit[1])

        unclamped = (
            self.gains.kp * error
            + self.gains.ki * candidate_integral
            + self.gains.kd * derivative
        )
        output = unclamped
        if self.output_limit is not None:
            output = clamp(output, self.output_limit[0], self.output_limit[1])

        # Accept the new integral when unsaturated, or when it helps move the
        # saturated output back toward the allowed range.
        if output == unclamped or (unclamped > output and error < 0.0) or (unclamped < output and error > 0.0):
            self.integral = candidate_integral

        self.previous_error = error
        return output

class RearWheelSpeedController:
    """Cascaded vx -> slip ratio -> rear wheel speed controller."""

    def __init__(
        self,
        params: TwinTrackParams | None = None,
        speed_gains: PIDGains = PIDGains(kp=0.18, ki=0.0),
        wheel_gains: PIDGains = PIDGains(kp=5, ki=0.01, kd=0.0),
        slip_limit: float = 0.20,
        torque_limit: float = 0.05,
    ) -> None:
        self.params = params or TwinTrackParams()
        self.slip_limit = abs(slip_limit)
        self.torque_limit = abs(torque_limit)
        self.speed_pi = PIDController(
            speed_gains,
            output_limit=(-self.slip_limit, self.slip_limit),
            integrator_limit=(-self.slip_limit, self.slip_limit),
        )
        self.left_wheel_pid = PIDController(
            wheel_gains,
            output_limit=(-self.torque_limit, self.torque_limit),
            integrator_limit=(-self.torque_limit, self.torque_limit),
        )
        self.right_wheel_pid = PIDController(
            wheel_gains,
            output_limit=(-self.torque_limit, self.torque_limit),
            integrator_limit=(-self.torque_limit, self.torque_limit),
        )

    def reset(self) -> None:
        self.speed_pi.reset()
        self.left_wheel_pid.reset()
        self.right_wheel_pid.reset()

    def step(self, state: "TwinTrackState", vx_ref: float, dt: float) -> RearTorqueCommand:
        vx_error = vx_ref - state.vx
        slip_ratio_ref = self.speed_pi.step(vx_error, dt)

        speed_sign = 1.0 if vx_ref >= 0.0 else -1.0
        speed_for_reference = max(abs(state.vx), self.params.st_min_velocity) * speed_sign
        omega_ref = speed_for_reference * (1.0 + slip_ratio_ref) / self.params.rear_wheel_radius

        left_torque = self.left_wheel_pid.step(omega_ref - state.omega_rl, dt)
        right_torque = self.right_wheel_pid.step(omega_ref - state.omega_rr, dt)

        return RearTorqueCommand(
            left_torque=left_torque,
            right_torque=right_torque,
            slip_ratio_ref=slip_ratio_ref,
            omega_ref_left=omega_ref,
            omega_ref_right=omega_ref,
        )

class RearWheelSpeedControllerLockedDiff:
    """
    Cascaded vx -> slip ratio -> rear wheel speed controller.
    This version treats both back wheels as one, essentially locking the differential
    """

    def __init__(
        self,
        tracking: str = "vx",
        params: TwinTrackParams | None = None,
        speed_gains: PIDGains = PIDGains(kp=0.18, ki=0.0),
        omega_gains: PIDGains = PIDGains(kp= 0.18, ki=0.1),
        wheel_gains: PIDGains = PIDGains(kp=5, ki=0.01, kd=0.0),
        slip_limit: float = 0.20,
        torque_limit: float = 0.05,
    ) -> None:
        self.params = params or TwinTrackParams()
        self.slip_limit = abs(slip_limit)
        self.torque_limit = abs(torque_limit)
        self.tracking = tracking
        if tracking == "vx":
            self.outer_pid = PIDController(
                speed_gains,
                output_limit=(-self.slip_limit, self.slip_limit),
                integrator_limit=(-self.slip_limit, self.slip_limit),
            )
        elif tracking == "omega_r":
            self.outer_pid = PIDController(
                omega_gains,
                output_limit=(-self.slip_limit, self.slip_limit),
                integrator_limit=(-self.slip_limit, self.slip_limit),
            )
        else:
            raise ValueError("Invalid tracking request:\nChoose either 'vx' or 'omega_r', depending on what reference you want the wheels to track")
        
        self.wheels_pid = PIDController(
            wheel_gains,
            output_limit=(-self.torque_limit, self.torque_limit),
            integrator_limit=(-self.torque_limit, self.torque_limit),
        )
        

    def reset(self) -> None:
        self.outer_pid.reset()
        self.wheels_pid.reset()

    def step(self, state: "TwinTrackState", ref: float, dt: float) -> RearTorqueCommand:

        if self.tracking == "vx":
            vx_error = ref - state.vx
            slip_ratio_ref = self.outer_pid.step(vx_error, dt)

            speed_sign = 1.0 if ref >= 0.0 else -1.0
            speed_for_reference = max(abs(state.vx), self.params.st_min_velocity) * speed_sign
            omega_ref = speed_for_reference * (1.0 + slip_ratio_ref) / self.params.rear_wheel_radius

        elif self.tracking == "omega_r":
            slip_ratio_ref = 0
            omega_ref = ref
        else:
            raise ValueError("Idk how you got here but you shouldn't be here...")

        mean_omega = (state.omega_rl + state.omega_rr) / 2
        rear_wheels_torque = self.wheels_pid.step(omega_ref - mean_omega, dt)

        return RearTorqueCommand(
            left_torque=rear_wheels_torque,
            right_torque=rear_wheels_torque,
            slip_ratio_ref=slip_ratio_ref,
            omega_ref_left=omega_ref,
            omega_ref_right=omega_ref,
        )

class TwinTrackModel:
    """Planar 4-wheel vehicle dynamics with Pacejka tire forces."""

    def __init__(self, params: TwinTrackParams | None = None) -> None:
        self.params = params or TwinTrackParams()
        self.max_steer = radians(self.params.max_steer_deg)

    def derivatives(
        self,
        state: TwinTrackState,
        steer: float = 0.0,
        rear_drive_torque: float = 0.0,
        front_brake_torque: float = 0.0,
        rear_brake_torque: float = 0.0,
        rear_left_torque: float | None = None,
        rear_right_torque: float | None = None,
        wheel_inputs: TwinTrackInput | None = None,
        drive_torque_fl: float | None = None,
        drive_torque_fr: float | None = None,
        drive_torque_rl: float | None = None,
        drive_torque_rr: float | None = None,
        brake_torque_fl: float | None = None,
        brake_torque_fr: float | None = None,
        brake_torque_rl: float | None = None,
        brake_torque_rr: float | None = None,
        steering_angle_fl: float | None = None,
        steering_angle_fr: float | None = None,
        steering_angle_rl: float | None = None,
        steering_angle_rr: float | None = None,
    ) -> tuple[TwinTrackState, dict[str, WheelForces]]:
        """Return continuous-time state derivatives.

        Args:
            state: Current vehicle state.
            steer: Compatibility front steering angle in radians.
            rear_drive_torque: Compatibility total drive torque for rear axle.
            front_brake_torque: Compatibility total front axle brake torque.
            rear_brake_torque: Compatibility total rear axle brake torque.
            rear_left_torque: Compatibility signed left rear drive torque.
            rear_right_torque: Compatibility signed right rear drive torque.
            wheel_inputs: Full per-wheel drive, brake, and steering inputs.
        """

        p = self.params
        inputs = self._resolve_inputs(
            steer=steer,
            rear_drive_torque=rear_drive_torque,
            front_brake_torque=front_brake_torque,
            rear_brake_torque=rear_brake_torque,
            rear_left_torque=rear_left_torque,
            rear_right_torque=rear_right_torque,
            wheel_inputs=wheel_inputs,
            drive_torque_fl=drive_torque_fl,
            drive_torque_fr=drive_torque_fr,
            drive_torque_rl=drive_torque_rl,
            drive_torque_rr=drive_torque_rr,
            brake_torque_fl=brake_torque_fl,
            brake_torque_fr=brake_torque_fr,
            brake_torque_rl=brake_torque_rl,
            brake_torque_rr=brake_torque_rr,
            steering_angle_fl=steering_angle_fl,
            steering_angle_fr=steering_angle_fr,
            steering_angle_rl=steering_angle_rl,
            steering_angle_rr=steering_angle_rr,
        )

        wheels = {
            "fl": (p.front_axle_len, p.front_track_width / 2.0, inputs.steering_angle_fl, p.front_wheel_radius),
            "fr": (p.front_axle_len, -p.front_track_width / 2.0, inputs.steering_angle_fr, p.front_wheel_radius),
            "rl": (-p.rear_axle_len, p.rear_track_width / 2.0, inputs.steering_angle_rl, p.rear_wheel_radius),
            "rr": (-p.rear_axle_len, -p.rear_track_width / 2.0, inputs.steering_angle_rr, p.rear_wheel_radius),
        }
        omegas = {
            "fl": state.omega_fl,
            "fr": state.omega_fr,
            "rl": state.omega_rl,
            "rr": state.omega_rr,
        }

        wheelbase = p.front_axle_len + p.rear_axle_len
        static_front_load = p.mass * p.gravity * p.rear_axle_len / wheelbase
        static_rear_load = p.mass * p.gravity * p.front_axle_len / wheelbase
        normal = {
            "fl": static_front_load / 2.0,
            "fr": static_front_load / 2.0,
            "rl": static_rear_load / 2.0,
            "rr": static_rear_load / 2.0,
        }
        # maybe room for improvement, nvm jestli delaji dynamic shift Fz?

        drive_torque = {
            "fl": inputs.drive_torque_fl,
            "fr": inputs.drive_torque_fr,
            "rl": inputs.drive_torque_rl,
            "rr": inputs.drive_torque_rr,
        }
        brake_torque = {
            "fl": inputs.brake_torque_fl,
            "fr": inputs.brake_torque_fr,
            "rl": inputs.brake_torque_rl,
            "rr": inputs.brake_torque_rr,
        }

        total_fx_body = 0.0
        total_fy_body = 0.0
        total_mz = 0.0
        omega_dot: dict[str, float] = {}
        force_debug: dict[str, WheelForces] = {}

        for name, (wx, wy, wheel_steer, radius) in wheels.items():
            v_wx_body = state.vx - state.yaw_rate * wy
            v_wy_body = state.vy + state.yaw_rate * wx

            c = cos(wheel_steer)
            s = sin(wheel_steer)
            v_long = c * v_wx_body + s * v_wy_body
            v_lat = -s * v_wx_body + c * v_wy_body
            wheel_circ_speed = radius * omegas[name]
            speed_ref = max(abs(v_long), abs(wheel_circ_speed), p.st_min_velocity)

            slip_ratio = (wheel_circ_speed - v_long) / speed_ref
            slip_angle = atan2(v_lat, speed_ref)

            tire_limit = p.tire_friction_coeff * normal[name]
            fx_tire_0 = pacejka_force(
                slip_ratio,
                tire_limit,
                p.pacejka_bx,
                p.pacejka_cx,
                p.pacejka_ex,
            )
            fy_tire_0 = -pacejka_force(
                slip_angle,
                tire_limit,
                p.pacejka_by,
                p.pacejka_cy,
                p.pacejka_ey,
            )


            """ EMPIRICAL APPROACH """

            
            # B_gx = 1
            # C_gx = 1
            # D_gx = 1
            # G_x = empirical_shaping(
            #     slip_angle,
            #     D_gx,
            #     B_gx,
            #     C_gx,
            # )

            # B_gy = 1
            # C_gy = 1
            # D_gy = 1
            # G_y = empirical_shaping(
            #     slip_ratio,
            #     D_gy,
            #     B_gy,
            #     C_gy,
            # )
            # fx_tire, fy_tire = G_x * fx_tire_0, G_y * fy_tire_0

            

            """ TRACTION ELLIPSE APPROACH """

            
            fx_tire, fy_tire = traction_ellipse_limit(
                fx_tire_0,
                fy_tire_0,
                tire_limit,
                tire_limit,
            )
        
            

            fx_body = c * fx_tire - s * fy_tire
            fy_body = s * fx_tire + c * fy_tire

            total_fx_body += fx_body
            total_fy_body += fy_body
            total_mz += wx * fy_body - wy * fx_body

            rolling = p.rolling_resistance * normal[name] * radius * sign_nonzero(omegas[name])
            braking = brake_torque[name] * sign_nonzero(omegas[name])
            omega_dot[name] = (drive_torque[name] - braking - radius * fx_tire - rolling) / p.wheel_inertia

            force_debug[name] = WheelForces(
                fx=fx_tire,
                fy=fy_tire,
                normal_force=normal[name],
                slip_ratio=slip_ratio,
                slip_angle=slip_angle,
            )

        x_dot = state.vx * cos(state.yaw) - state.vy * sin(state.yaw)
        y_dot = state.vx * sin(state.yaw) + state.vy * cos(state.yaw)
        vx_dot = total_fx_body / p.mass + state.vy * state.yaw_rate
        vy_dot = total_fy_body / p.mass - state.vx * state.yaw_rate
        yaw_rate_dot = total_mz / p.moment_of_inertia

        return (
            TwinTrackState(
                x=x_dot,
                y=y_dot,
                yaw=state.yaw_rate,
                vx=vx_dot,
                vy=vy_dot,
                yaw_rate=yaw_rate_dot,
                omega_fl=omega_dot["fl"],
                omega_fr=omega_dot["fr"],
                omega_rl=omega_dot["rl"],
                omega_rr=omega_dot["rr"],
            ),
            force_debug,
        )

    def step(
        self,
        state: TwinTrackState,
        dt: float,
        steer: float = 0.0,
        rear_drive_torque: float = 0.0,
        front_brake_torque: float = 0.0,
        rear_brake_torque: float = 0.0,
        rear_left_torque: float | None = None,
        rear_right_torque: float | None = None,
        wheel_inputs: TwinTrackInput | None = None,
        drive_torque_fl: float | None = None,
        drive_torque_fr: float | None = None,
        drive_torque_rl: float | None = None,
        drive_torque_rr: float | None = None,
        brake_torque_fl: float | None = None,
        brake_torque_fr: float | None = None,
        brake_torque_rl: float | None = None,
        brake_torque_rr: float | None = None,
        steering_angle_fl: float | None = None,
        steering_angle_fr: float | None = None,
        steering_angle_rl: float | None = None,
        steering_angle_rr: float | None = None,
    ) -> tuple[TwinTrackState, TwinTrackStateDerivatives, dict[str, dict[str, float]]]:
        """Advance the model by one fourth-order Runge-Kutta integration step."""

        k1, k1_wh = self.derivatives(
            state,
            steer,
            rear_drive_torque,
            front_brake_torque,
            rear_brake_torque,
            rear_left_torque,
            rear_right_torque,
            wheel_inputs,
            drive_torque_fl,
            drive_torque_fr,
            drive_torque_rl,
            drive_torque_rr,
            brake_torque_fl,
            brake_torque_fr,
            brake_torque_rl,
            brake_torque_rr,
            steering_angle_fl,
            steering_angle_fr,
            steering_angle_rl,
            steering_angle_rr,
        )
        k2, k2_wh = self.derivatives(
            add_scaled_state(state, k1, 0.5 * dt),
            steer,
            rear_drive_torque,
            front_brake_torque,
            rear_brake_torque,
            rear_left_torque,
            rear_right_torque,
            wheel_inputs,
            drive_torque_fl,
            drive_torque_fr,
            drive_torque_rl,
            drive_torque_rr,
            brake_torque_fl,
            brake_torque_fr,
            brake_torque_rl,
            brake_torque_rr,
            steering_angle_fl,
            steering_angle_fr,
            steering_angle_rl,
            steering_angle_rr,
        )
        k3, k3_wh = self.derivatives(
            add_scaled_state(state, k2, 0.5 * dt),
            steer,
            rear_drive_torque,
            front_brake_torque,
            rear_brake_torque,
            rear_left_torque,
            rear_right_torque,
            wheel_inputs,
            drive_torque_fl,
            drive_torque_fr,
            drive_torque_rl,
            drive_torque_rr,
            brake_torque_fl,
            brake_torque_fr,
            brake_torque_rl,
            brake_torque_rr,
            steering_angle_fl,
            steering_angle_fr,
            steering_angle_rl,
            steering_angle_rr,
        )
        k4, k4_wh = self.derivatives(
            add_scaled_state(state, k3, dt),
            steer,
            rear_drive_torque,
            front_brake_torque,
            rear_brake_torque,
            rear_left_torque,
            rear_right_torque,
            wheel_inputs,
            drive_torque_fl,
            drive_torque_fr,
            drive_torque_rl,
            drive_torque_rr,
            brake_torque_fl,
            brake_torque_fr,
            brake_torque_rl,
            brake_torque_rr,
            steering_angle_fl,
            steering_angle_fr,
            steering_angle_rl,
            steering_angle_rr,
        )
        return add_rk4_state(state, k1, k2, k3, k4, k1_wh, k2_wh, k3_wh, k4_wh, dt)

    def _resolve_inputs(
        self,
        steer: float,
        rear_drive_torque: float,
        front_brake_torque: float,
        rear_brake_torque: float,
        rear_left_torque: float | None,
        rear_right_torque: float | None,
        wheel_inputs: TwinTrackInput | None,
        drive_torque_fl: float | None,
        drive_torque_fr: float | None,
        drive_torque_rl: float | None,
        drive_torque_rr: float | None,
        brake_torque_fl: float | None,
        brake_torque_fr: float | None,
        brake_torque_rl: float | None,
        brake_torque_rr: float | None,
        steering_angle_fl: float | None,
        steering_angle_fr: float | None,
        steering_angle_rl: float | None,
        steering_angle_rr: float | None,
    ) -> TwinTrackInput:
        if rear_left_torque is None and rear_right_torque is None:
            compat_drive_rl = rear_drive_torque / 2.0
            compat_drive_rr = rear_drive_torque / 2.0
        elif rear_left_torque is None or rear_right_torque is None:
            raise ValueError("rear_left_torque and rear_right_torque must be provided together")
        else:
            compat_drive_rl = rear_left_torque
            compat_drive_rr = rear_right_torque

        base = wheel_inputs or TwinTrackInput(
            drive_torque_rl=compat_drive_rl,
            drive_torque_rr=compat_drive_rr,
            brake_torque_fl=front_brake_torque / 2.0,
            brake_torque_fr=front_brake_torque / 2.0,
            brake_torque_rl=rear_brake_torque / 2.0,
            brake_torque_rr=rear_brake_torque / 2.0,
            steering_angle_fl=steer,
            steering_angle_fr=steer,
        )
        # nehandluje separate brake torques na zadnich kolech, coz bude u torque vectoringu potreba dodelat
        

        return TwinTrackInput(
            drive_torque_fl=value_or(base.drive_torque_fl, drive_torque_fl),
            drive_torque_fr=value_or(base.drive_torque_fr, drive_torque_fr),
            drive_torque_rl=value_or(base.drive_torque_rl, drive_torque_rl),
            drive_torque_rr=value_or(base.drive_torque_rr, drive_torque_rr),
            brake_torque_fl=value_or(base.brake_torque_fl, brake_torque_fl),
            brake_torque_fr=value_or(base.brake_torque_fr, brake_torque_fr),
            brake_torque_rl=value_or(base.brake_torque_rl, brake_torque_rl),
            brake_torque_rr=value_or(base.brake_torque_rr, brake_torque_rr),
            steering_angle_fl=clamp(value_or(base.steering_angle_fl, steering_angle_fl), -self.max_steer, self.max_steer),
            steering_angle_fr=clamp(value_or(base.steering_angle_fr, steering_angle_fr), -self.max_steer, self.max_steer),
            steering_angle_rl=clamp(value_or(base.steering_angle_rl, steering_angle_rl), -self.max_steer, self.max_steer),
            steering_angle_rr=clamp(value_or(base.steering_angle_rr, steering_angle_rr), -self.max_steer, self.max_steer),
        )


def pacejka_force(slip: float, peak_force: float, b: float, c: float, e: float) -> float:
    """Magic-formula tire force for one slip dimension."""

    shaped_slip = b * slip
    return peak_force * sin(c * atan(shaped_slip - e * (shaped_slip - atan(shaped_slip))))

def empirical_shaping(slip: float, d: float, b: float, c: float) -> float:

    shaped_slip = b * slip
    return d * cos(c * atan(shaped_slip))

def traction_ellipse_limit(fx: float, fy: float, fx_limit: float, fy_limit: float) -> tuple[float, float]:
    """Limit combined tire force with a traction ellipse."""

    if fx_limit <= 0.0 or fy_limit <= 0.0:
        return 0.0, 0.0

    ellipse_value = (fx / fx_limit) ** 2 + (fy / fy_limit) ** 2
    if ellipse_value <= 1.0 or ellipse_value <= 1e-12: # ???
        return fx, fy

    scale = 1.0 / (ellipse_value ** 0.5)
    # not sure jestli je tohle dobre uplne ale oka
    return fx * scale, fy * scale


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def value_or(default: float, override: float | None) -> float:
    if override is None:
        return default
    return override


def sign_nonzero(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def add_scaled_state(state: TwinTrackState, derivative: TwinTrackState, scale: float) -> TwinTrackState:
    return TwinTrackState(
        x=state.x + scale * derivative.x,
        y=state.y + scale * derivative.y,
        yaw=state.yaw + scale * derivative.yaw,
        vx=state.vx + scale * derivative.vx,
        vy=state.vy + scale * derivative.vy,
        yaw_rate=state.yaw_rate + scale * derivative.yaw_rate,
        omega_fl=state.omega_fl + scale * derivative.omega_fl,
        omega_fr=state.omega_fr + scale * derivative.omega_fr,
        omega_rl=state.omega_rl + scale * derivative.omega_rl,
        omega_rr=state.omega_rr + scale * derivative.omega_rr,
    )

def _rk4_avg(v1: float, v2: float, v3: float, v4: float) -> float:
    """Spočítá vážený průměr RK4 pro 4 skalarne hodnoty."""
    return (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0


def _avg_wheel(
    w1: WheelForces, w2: WheelForces, w3: WheelForces, w4: WheelForces
) -> dict[str, float]:
    attrs = ["fx", "fy", "normal_force", "slip_ratio", "slip_angle"]
    return {
        attr: _rk4_avg(
            getattr(w1, attr),
            getattr(w2, attr),
            getattr(w3, attr),
            getattr(w4, attr),
        )
        for attr in attrs
    }


def combined_tire_params(
    k1_wh: dict[str, WheelForces],
    k2_wh: dict[str, WheelForces],
    k3_wh: dict[str, WheelForces],
    k4_wh: dict[str, WheelForces],
) -> dict[str, dict[str, float]]:
    return {
        w: _avg_wheel(k1_wh[w], k2_wh[w], k3_wh[w], k4_wh[w])
        for w in ["fl", "fr", "rl", "rr"]
    }


def add_rk4_state(
    state: TwinTrackState,
    k1: TwinTrackState,
    k2: TwinTrackState,
    k3: TwinTrackState,
    k4: TwinTrackState,
    k1_wh: dict[str, WheelForces],
    k2_wh: dict[str, WheelForces],
    k3_wh: dict[str, WheelForces],
    k4_wh: dict[str, WheelForces],
    dt: float,
) -> tuple[TwinTrackState, TwinTrackStateDerivatives, dict[str, dict[str, float]]]:
    
    # 1. Automatický výpočet všech derivací pro jednotlivá pole TwinTrackState
    state_fields = [f.name for f in fields(TwinTrackState)]
    
    derivatives_dict = {
        f"{field}_dot": _rk4_avg(
            getattr(k1, field),
            getattr(k2, field),
            getattr(k3, field),
            getattr(k4, field),
        )
        for field in state_fields
    }

    # 2. Nový stav (state + dt * derivace)
    new_state_kwargs = {
        field: getattr(state, field) + dt * derivatives_dict[f"{field}_dot"]
        for field in state_fields
    }
    result_state = TwinTrackState(**new_state_kwargs)

    # 3. Objekt s použitými derivacemi
    used_derivatives = TwinTrackStateDerivatives(**derivatives_dict)

    # 4. Spojení a rozšíření parametrů kol bez opakování kódu
    wheel_forces = combined_tire_params(k1_wh, k2_wh, k3_wh, k4_wh)

    for w in ["fl", "fr", "rl", "rr"]:
        wheel_forces[w]["omega"] = getattr(state, f"omega_{w}")
        wheel_forces[w]["omega_dot"] = derivatives_dict[f"omega_{w}_dot"]

    return result_state, used_derivatives, wheel_forces


def run_controller_self_test() -> TwinTrackState:
    model = TwinTrackModel()
    controller = RearWheelSpeedController(model.params, torque_limit=0.2)
    initial_omega = 0.2 / model.params.rear_wheel_radius
    state = TwinTrackState(
        vx=0.2,
        omega_fl=initial_omega,
        omega_fr=initial_omega,
        omega_rl=initial_omega,
        omega_rr=initial_omega,
    )
    initial_vx = state.vx
    max_abs_torque = 0.0
    max_abs_slip_ref = 0.0

    for _ in range(300):
        command = controller.step(state, vx_ref=1.0, dt=0.01)
        max_abs_torque = max(max_abs_torque, abs(command.left_torque), abs(command.right_torque))
        max_abs_slip_ref = max(max_abs_slip_ref, abs(command.slip_ratio_ref))
        state, derivatives, wheel_forces = model.step(
            state,
            dt=0.01,
            steer=0.0,
            rear_left_torque=command.left_torque,
            rear_right_torque=command.right_torque,
        )

    values = (
        state.x,
        state.y,
        state.yaw,
        state.vx,
        state.vy,
        state.yaw_rate,
        state.omega_rl,
        state.omega_rr,
        max_abs_torque,
        max_abs_slip_ref,
    )
    if not all(isfinite(value) for value in values):
        raise RuntimeError("controller self-test produced a non-finite value")
    if state.vx <= initial_vx:
        raise RuntimeError("controller self-test did not increase vx")
    if max_abs_torque > controller.torque_limit + 1e-12:
        raise RuntimeError("controller self-test exceeded torque limit")
    if max_abs_slip_ref > controller.slip_limit + 1e-12:
        raise RuntimeError("controller self-test exceeded slip limit")

    return state


if __name__ == "__main__":
    model = TwinTrackModel()
    state = TwinTrackState(vx=0.5)

    for _ in range(200):
        state, derivatives, wheel_forces = model.step(
            state,
            dt=0.01,
            steer=radians(10.0),
            rear_drive_torque=0.015,
        )

    print(state)
    print(run_controller_self_test())
