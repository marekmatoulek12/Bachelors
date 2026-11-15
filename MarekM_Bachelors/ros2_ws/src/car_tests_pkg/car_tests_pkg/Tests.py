import signal

import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions

from .Blueprints import LoggingConnectionBP, PIDController, PIDGains


class InitTest(LoggingConnectionBP):
    def __init__(self, node_name: str = "init_test", test_length: float = 20.0, timer_period: float = 0.1):
        super().__init__(
            node_name=node_name,
            control_loop=self.control_loop,
            timer_period=timer_period,
            use_fw_encoders=False,
        )
        self.test_length = test_length
        self.timer_period = timer_period

        self.start_time = self.get_clock().now().nanoseconds / 1e9

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time
        rpm_req = 0.0
        steering_angle_req_deg = 0.0

        if elapsed_time >= self.test_length:
            self.get_logger().info("Test length exceeded. Stopping test.")
            return False

        # Logovani pozadavku (rizeni ve stupnich)
        self.log_commands(rpm_req, steering_angle_req_deg)

        # Publikovani pozadavku
        self.publish_data()

        # Logovani do CSV
        self.log_data()

        return True

class SineWaveTest(LoggingConnectionBP):
    def __init__(
        self,
        node_name: str = "sine_wave_test",
        target_speed_m_s: float = 1.0,
        sine_freq: float = 1.0,
        sine_ampl_deg: float = 5.0,
        test_length: float = 20.0,
        timer_period: float = 0.1,
    ):
        super().__init__(
            node_name=node_name,
            control_loop=self.control_loop,
            timer_period=timer_period,
        )
        self.target_speed_m_s = target_speed_m_s  # m/s
        self.sine_freq = sine_freq                # rad/s
        self.sine_ampl_deg = sine_ampl_deg        # deg
        self.timer_period = timer_period          # s
        self.test_length = test_length            # s

        self.speed2rpm_controller = PIDController(
            gains=PIDGains(kp=70, ki=150, kd=0),
            output_limit=(-200, 200),
            integrator_limit=(-200, 200),
        )

        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.test_phase = 1
        self.ph_1_end_time = 0.0

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time
        rpm_req = 0.0
        steering_angle_req_deg = 0.0

        if elapsed_time >= self.test_length:
            self.get_logger().info("Test length exceeded. Stopping test.")
            return False

        if self.all_data_available():
            match self.test_phase:
                case 1:
                    # Faze 1: Zrychleni na cilovou rychlost
                    rpm_ff = self.vx2rpm(self.target_speed_m_s)
                    rpm_fb = self.speed2rpm_controller.step(self.target_speed_m_s - self.vx, self.timer_period)  # type: ignore
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req_deg = 0.0

                    self.get_logger().info(f"Zrychluji... Aktualni rychlost: {self.vx:.2f} m/s")
                    if self.vx > self.target_speed_m_s * 0.95:  # type: ignore
                        self.get_logger().info(f"Dosazena dostatecna rychlost {self.vx:.2f} :: FAZE 2")
                        self.test_phase = 2
                        self.ph_1_end_time = current_time

                case 2:
                    # Faze 2: Sinusove rizeni pri udrzovani cilove rychlosti
                    time_from_ph_1 = current_time - self.ph_1_end_time

                    rpm_ff = self.vx2rpm(self.target_speed_m_s)
                    rpm_fb = self.speed2rpm_controller.step(self.target_speed_m_s - self.vx, self.timer_period)  # type: ignore
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req_deg = self.sine_ampl_deg * np.sin(self.sine_freq * time_from_ph_1)
                    self.get_logger().info(f"Zatacim: {steering_angle_req_deg:.2f} deg")

            self.log_commands(rpm_req, steering_angle_req_deg)
            self.publish_data()
            self.log_data()

        return True

class RampSteerTest(LoggingConnectionBP):
    def __init__(
        self,
        node_name: str = "ramp_steer_test",
        target_speed_m_s: float = 10.0,
        steering_angle_rate_deg_s: float = 1.0,
        test_length: float = 20.0,
        timer_period: float = 0.1,
    ):
        super().__init__(
            node_name=node_name,
            control_loop=self.control_loop,
            timer_period=timer_period,
        )
        self.target_speed_m_s = target_speed_m_s          # m/s
        self.steer_rate_deg_s = steering_angle_rate_deg_s # deg/s
        self.timer_period = timer_period                  # s
        self.test_length = test_length                    # s

        self.speed2rpm_controller = PIDController(
            gains=PIDGains(kp=70, ki=150, kd=0),
            output_limit=(-200, 200),
            integrator_limit=(-200, 200),
        )

        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.test_phase = 1
        self.ph_1_end_time = 0.0

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time
        rpm_req = 0.0
        steering_angle_req_deg = 0.0

        if elapsed_time >= self.test_length:
            self.get_logger().info("Test length exceeded. Stopping test.")
            return False

        if self.all_data_available():
            match self.test_phase:
                case 1:
                    # Faze 1: Zrychleni na cilovou rychlost
                    rpm_ff = self.vx2rpm(self.target_speed_m_s)
                    rpm_fb = self.speed2rpm_controller.step(self.target_speed_m_s - self.vx, self.timer_period)  # type: ignore
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req_deg = 0.0

                    self.get_logger().info(f"Zrychluji... Aktualni rychlost: {self.vx:.2f} m/s")
                    if self.vx > self.target_speed_m_s * 0.95:  # type: ignore
                        self.get_logger().info(f"Dosazena dostatecna rychlost {self.vx:.2f} :: FAZE 2")
                        self.test_phase = 2
                        self.ph_1_end_time = current_time

                case 2:
                    # Faze 2: Plynule zvetsovani uhlu zatoceni ve stupnich
                    time_from_ph_1 = current_time - self.ph_1_end_time

                    rpm_ff = self.vx2rpm(self.target_speed_m_s)
                    rpm_fb = self.speed2rpm_controller.step(self.target_speed_m_s - self.vx, self.timer_period)  # type: ignore
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req_deg = self.steer_rate_deg_s * time_from_ph_1
                    self.get_logger().info(f"Zatacim: {steering_angle_req_deg:.2f} deg")

            self.log_commands(rpm_req, steering_angle_req_deg)
            self.publish_data()
            self.log_data()

        return True

class RampSpeedTest(LoggingConnectionBP):
    def __init__(
        self,
        node_name: str = "ramp_speed_test",
        start_speed_m_s: float = 10.0,
        target_steer_deg: float = 5.0,
        acceleration_m_ss: float = 1.0,
        test_length: float = 20.0,
        timer_period: float = 0.1,
    ):
        super().__init__(
            node_name=node_name,
            control_loop=self.control_loop,
            timer_period=timer_period,
        )

        self.start_speed = start_speed_m_s
        self.target_steer_deg = target_steer_deg
        self.acceleration = acceleration_m_ss
        self.timer_period = timer_period  # s
        self.test_length = test_length    # s

        self.speed2rpm_controller = PIDController(
            gains=PIDGains(kp=70, ki=150, kd=0),
            output_limit=(-200, 200),
            integrator_limit=(-200, 200),
        )

        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.test_phase = 1
        self.ph_1_end_time = 0.0
        self.ph_2_end_time = 0.0

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time
        rpm_req = 0.0
        steering_angle_req_deg = 0.0

        if elapsed_time >= self.test_length:
            self.get_logger().info("Test length exceeded. Stopping test.")
            return False

        if self.all_data_available():
            match self.test_phase:
                case 1:
                    # Faze 1: Zrychleni na cilovou rychlost
                    tar_sp = self.start_speed
                    rpm_ff = self.vx2rpm(tar_sp)
                    rpm_fb = self.speed2rpm_controller.step(tar_sp - self.vx, self.timer_period)  # type: ignore
                    self.get_logger().info(f"Error 1: {(tar_sp - self.vx):.2f} m/s -- rpm fb: {rpm_fb:.2f}")
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req_deg = 0.0

                    if self.vx >= self.start_speed * 0.95:  # type: ignore
                        self.test_phase = 2
                        self.ph_1_end_time = current_time

                case 2:
                    # Faze 2: Drzet cilovou rychlost a zatocit na pozadovany uhel (ve stupnich)
                    phase_2_duration = 1.0
                    tar_sp = self.start_speed
                    rpm_ff = self.vx2rpm(tar_sp)
                    rpm_fb = self.speed2rpm_controller.step(tar_sp - self.vx, self.timer_period)  # type: ignore
                    self.get_logger().info(f"Error 2: {(tar_sp - self.vx):.2f} m/s -- rpm fb: {rpm_fb:.2f}")
                    rpm_req = rpm_ff + rpm_fb

                    time_from_ph_1 = current_time - self.ph_1_end_time
                    steering_angle_req_deg = self.target_steer_deg * min(time_from_ph_1 / phase_2_duration, 1.0)

                    if time_from_ph_1 >= phase_2_duration:
                        self.test_phase = 3
                        self.ph_2_end_time = current_time

                case 3:
                    # Faze 3: Plynule zrychlovat pri zachovani uhlu rizeni
                    time_from_ph_2 = current_time - self.ph_2_end_time
                    tar_sp = self.start_speed + self.acceleration * time_from_ph_2
                    rpm_ff = self.vx2rpm(tar_sp)
                    rpm_fb = self.speed2rpm_controller.step(tar_sp - self.vx, self.timer_period)  # type: ignore
                    self.get_logger().info(f"Error 3: {(tar_sp - self.vx):.2f} m/s -- rpm fb: {rpm_fb:.2f}")
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req_deg = self.target_steer_deg

            self.log_commands(rpm_req, steering_angle_req_deg)
            self.publish_data()
            self.log_data()

        return True


class CombinedSteerSpeedTest(LoggingConnectionBP):
    """Kombinovaný test (RampSteer + RampSpeed):
    1. Dosáhne počáteční rychlosti (starting_velocity)
    2. Natočí kola na výchozí úhel (target_steer_deg)
    3. Současně mění rychlost (acceleration_m_ss) i úhel řízení (steering_rate [deg/s])
    """

    def __init__(
        self,
        node_name: str = "combined_steer_speed_test",
        starting_velocity: float = 1.0,
        target_steer_deg: float = 0.0,
        steering_rate: float = 5.0,
        acceleration_m_ss: float = 1.0,
        test_length: float = 20.0,
        timer_period: float = 0.01,
    ):
        super().__init__(
            node_name=node_name,
            control_loop=self.control_loop,
            timer_period=timer_period,
        )

        self.starting_velocity = starting_velocity  # m/s
        self.target_steer_deg = target_steer_deg    # deg (výchozí úhel před rampou)
        self.steering_rate = steering_rate          # deg/s (kladné = doleva, záporné = doprava)
        self.acceleration = acceleration_m_ss       # m/s^2
        self.timer_period = timer_period            # s
        self.test_length = test_length              # s

        self.speed2rpm_controller = PIDController(
            gains=PIDGains(kp=70, ki=150, kd=0),
            output_limit=(-200, 200),
            integrator_limit=(-200, 200),
        )

        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.test_phase = 1
        self.ph_1_end_time = 0.0
        self.ph_2_end_time = 0.0

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time
        rpm_req = 0.0
        steering_angle_req_deg = 0.0

        if elapsed_time >= self.test_length:
            self.get_logger().info("Test length exceeded. Stopping test.")
            return False

        if self.all_data_available():
            match self.test_phase:
                case 1:
                    # Faze 1: Zrychleni na pocatecni rychlost
                    tar_sp = self.starting_velocity
                    rpm_ff = self.vx2rpm(tar_sp)
                    rpm_fb = self.speed2rpm_controller.step(tar_sp - self.vx, self.timer_period)  # type: ignore
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req_deg = 0.0

                    self.get_logger().info(
                        f"Faze 1 | v: {self.vx:.2f}/{tar_sp:.2f} m/s | steer: {steering_angle_req_deg:.2f} deg"
                    )

                    if self.starting_velocity <= 0.0 or self.vx >= self.starting_velocity * 0.95:  # type: ignore
                        self.test_phase = 2
                        self.ph_1_end_time = current_time

                case 2:
                    # Faze 2: Udrzet pocatecni rychlost a nastavit vychozi uhel zatoceni
                    phase_2_duration = 1.0
                    tar_sp = self.starting_velocity
                    rpm_ff = self.vx2rpm(tar_sp)
                    rpm_fb = self.speed2rpm_controller.step(tar_sp - self.vx, self.timer_period)  # type: ignore
                    rpm_req = rpm_ff + rpm_fb

                    if abs(self.target_steer_deg) < 1e-6:
                        steering_angle_req_deg = 0.0
                        self.test_phase = 3
                        self.ph_2_end_time = current_time
                    else:
                        time_from_ph_1 = current_time - self.ph_1_end_time
                        steering_angle_req_deg = self.target_steer_deg * min(
                            time_from_ph_1 / phase_2_duration, 1.0
                        )
                        self.get_logger().info(
                            f"Faze 2 | v: {self.vx:.2f} m/s | steer: {steering_angle_req_deg:.2f}/{self.target_steer_deg:.2f} deg"
                        )
                        if time_from_ph_1 >= phase_2_duration:
                            self.test_phase = 3
                            self.ph_2_end_time = current_time

                case 3:
                    # Faze 3: Soucasna zmena rychlosti i uhlu zatoceni
                    time_from_ph_2 = current_time - self.ph_2_end_time
                    tar_sp = self.starting_velocity + self.acceleration * time_from_ph_2
                    steering_angle_req_deg = self.target_steer_deg + self.steering_rate * time_from_ph_2

                    rpm_ff = self.vx2rpm(tar_sp)
                    rpm_fb = self.speed2rpm_controller.step(tar_sp - self.vx, self.timer_period)  # type: ignore
                    rpm_req = rpm_ff + rpm_fb

                    self.get_logger().info(
                        f"Faze 3 | tar_v: {tar_sp:.2f} (act: {self.vx:.2f}) m/s | steer: {steering_angle_req_deg:.2f} deg"
                    )

            # Logovani pozadavku (steering ve stupnich)
            self.log_commands(rpm_req, steering_angle_req_deg)

            # Publikovani pozadavku (prepocet na radiany resi Blueprints)
            self.publish_data()

            # Logovani do CSV
            self.log_data()

        return True

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    def sgn_str(x: float) -> str:
        return "" if x >= 0 else "_"

    """ INIT TEST SETUP """
    # test_length = 20.0  # seconds
    # timer_period = 0.05  # 20 Hz
    # node = InitTest(
    #     node_name="first_test_eva",
    #     test_length=test_length,
    #     timer_period=timer_period,
    # )

    """ SINE WAVE TEST SETUP """
    # target_speed_m_s = 1.0
    # sine_freq = 1.0
    # sine_ampl_deg = 10.0
    # test_length = 10.0
    # timer_period = 0.01
    # node = SineWaveTest(
    #     target_speed_m_s=target_speed_m_s,
    #     sine_freq=sine_freq,
    #     sine_ampl_deg=sine_ampl_deg,
    #     test_length=test_length,
    #     timer_period=timer_period,
    # )

    """ RAMP STEER TEST SETUP """
    # target_speed_m_s = 10.0  # m/s
    # steering_rate = 1.0  # deg/s
    # test_length = 20.0  # seconds
    # timer_period = 0.1  # 10 Hz
    # node = RampSteerTest(
    #     target_speed_m_s=target_speed_m_s,
    #     steering_angle_rate_deg_s=steering_rate,
    #     test_length=test_length,
    #     timer_period=timer_period,
    # )

    """ RAMP SPEED TEST SETUP """
    start_speed_m_s = 1.0
    target_steer_deg = 0.0
    acceleration_m_ss = 0.0
    test_length = 20.0  # seconds
    timer_period = 0.01  # 100 Hz
    node = RampSpeedTest(
        node_name=f"rst_{sgn_str(target_steer_deg)}{int(abs(target_steer_deg))}_{int(acceleration_m_ss)}",
        start_speed_m_s=start_speed_m_s,
        target_steer_deg=target_steer_deg,
        acceleration_m_ss=acceleration_m_ss,
        test_length=test_length,
        timer_period=timer_period,
    )

    """ COMBINED STEER & SPEED TEST SETUP """
    # starting_velocity = 1.0    # m/s
    # target_steer_deg = 0.0     # deg
    # steering_rate = 5.0        # deg/s
    # acceleration_m_ss = 1.0    # m/s^2
    # test_length = 20.0         # seconds
    # timer_period = 0.01        # 100 Hz

    # node = CombinedSteerSpeedTest(
    #     node_name=f"csst_{sgn_str(target_steer_deg)}{int(abs(target_steer_deg))}_{int(steering_rate)}_{int(starting_velocity)}_{int(acceleration_m_ss)}",
    #     starting_velocity=starting_velocity,
    #     target_steer_deg=target_steer_deg,
    #     steering_rate=steering_rate,
    #     acceleration_m_ss=acceleration_m_ss,
    #     test_length=test_length,
    #     timer_period=timer_period,
    # )

    """ SPUSTENI TESTU """
    stop_requested = False
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while rclpy.ok() and not node.should_stop and not stop_requested:
            rclpy.spin_once(node, timeout_sec=timer_period)
    finally:
        try:
            if node is not None:
                try:
                    node.destroy_blueprint_node()
                except Exception:
                    pass
        finally:
            try:
                rclpy.shutdown()
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    main()