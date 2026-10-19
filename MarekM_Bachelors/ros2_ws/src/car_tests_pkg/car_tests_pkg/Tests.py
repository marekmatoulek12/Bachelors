import signal

import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions

from .Blueprints import TestBluePrint, PIDController, PIDGains

class InitTest(TestBluePrint):
    def __init__(self, node_name: str = "init_test", test_length: float = 20.0, timer_period: float = 0.1):
        super().__init__(node_name=node_name,
                         control_loop=self.control_loop,
                         timer_period=timer_period,
                         use_fw_encoders= False
                         )
        self.test_length = test_length
        self.timer_period = timer_period

        self.start_time = self.get_clock().now().nanoseconds / 1e9

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time
        rpm_req = 0.0
        steering_angle_req = 0.0

        if elapsed_time >= self.test_length:
            self.get_logger().info("Test length exceeded. Stopping test.")
            return False

        # Logovani pozadavku
        self.log_commands(rpm_req, steering_angle_req)

        # Publikovani pozadavku
        self.publish_data(rpm_req, steering_angle_req)

        # Logovani do CSV
        self.log_data(rpm_req, steering_angle_req)

        return True

class SineWaveTest(TestBluePrint):
    def __init__(self, node_name: str = "sine_wave_test", target_speed_m_s: float = 1.0, sine_freq: float = 1.0, sine_ampl: float = 1.0, test_length: float = 20.0, timer_period: float = 0.1):
        super().__init__(node_name=node_name,
                         control_loop=self.control_loop,
                         timer_period=timer_period
                         )
        self.target_speed_m_s = target_speed_m_s  # m/s
        self.sine_freq = sine_freq
        self.sine_ampl = sine_ampl
        self.timer_period = timer_period          # seconds
        self.test_length = test_length      # seconds
    
    
        ## ROOM FOR IMPROVEMENT -- KONSTANTY NATRELENY Z PRDELE
        self.speed2rpm_controller = PIDController(
            gains=PIDGains(kp=0, ki=0.1, kd=0),
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
        steering_angle_req = 0.0
        
        if elapsed_time >= self.test_length:
            self.get_logger().info("Test length exceeded. Stopping test.")
            return False
    
        if self.all_data_available():
            match self.test_phase:
    
                case 1:
                    # Faze 1: Zrychleni na cilovou rychlost
                    rpm_ff = self.vx2rpm(self.target_speed_m_s)
                    rpm_fb = self.speed2rpm_controller.step(self.target_speed_m_s - self.vx, self.timer_period) # type: ignore
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req = 0.0
    
                    self.get_logger().info(f"Zrychluji... Aktualni rychlost: {self.vx:.2f} m/s")
                    if self.vx > self.target_speed_m_s * 0.95: # type: ignore
                        self.get_logger().info(f"Dosazena dostatecna rychlost {self.vx} :: FAZE 2")
                        self.test_phase = 2
                        self.ph_1_end_time = current_time
    
                case 2:
                    # Faze 2: Pokud jsme zhruba dosahli cilove rychlosti, zacneme pomalu zvetsovat uhel zatoceni
                    time_from_ph_1 = current_time - self.ph_1_end_time
    
                    rpm_ff = self.vx2rpm(self.target_speed_m_s)
                    rpm_fb = self.speed2rpm_controller.step(self.target_speed_m_s - self.vx, self.timer_period) # type: ignore
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req = self.sine_ampl * np.sin(self.sine_freq * time_from_ph_1) 
                    self.get_logger().info(f"Zatacim: {steering_angle_req:.2f} deg")
    
            # Logovani pozadavku
            self.log_commands(rpm_req, steering_angle_req)
    
            # Publikovani pozadavku
            self.publish_data(rpm_req, steering_angle_req)
    
            # Logovani do CSV
            self.log_data(rpm_req, steering_angle_req)
    
        return True
    
        # ROS2 logovani
        # otevrit v druhem terminalu:
        # ros2 bag record -o muj_rampsteer_bag /state/odometry /servo/front/angle_req /powertrain/rl/rpm_req /powertrain/rr/rpm_req


class RampSteerTest(TestBluePrint):
    def __init__(self, node_name: str = "ramp_steer_test", target_speed_m_s: float = 10.0, steering_angle_rate_deg_s: float = 1.0, test_length: float = 20.0, timer_period: float = 0.1):
        super().__init__(node_name=node_name,
                         control_loop=self.control_loop,
                         timer_period=timer_period
                         )
        self.target_speed_m_s = target_speed_m_s  # m/s
        self.steer_rate_deg_s = steering_angle_rate_deg_s    # degrees per second
        self.timer_period = timer_period          # seconds
        self.test_length = test_length      # seconds


        ## ROOM FOR IMPROVEMENT -- KONSTANTY NATRELENY Z PRDELE
        self.speed2rpm_controller = PIDController(
            gains=PIDGains(kp=0, ki=0.1, kd=0),
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
        steering_angle_req = 0.0

        steer_rate_rad_s = self.deg2rad(self.steer_rate_deg_s)

        if elapsed_time >= self.test_length:
            self.get_logger().info("Test length exceeded. Stopping test.")
            return False

        if self.all_data_available():
            match self.test_phase:

                case 1:
                    # Faze 1: Zrychleni na cilovou rychlost
                    rpm_ff = self.vx2rpm(self.target_speed_m_s)
                    rpm_fb = self.speed2rpm_controller.step(self.target_speed_m_s - self.vx, self.timer_period) # type: ignore
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req = 0.0

                    self.get_logger().info(f"Zrychluji... Aktualni rychlost: {self.vx:.2f} m/s")
                    if self.vx > self.target_speed_m_s * 0.95: # type: ignore
                        self.get_logger().info(f"Dosazena dostatecna rychlost {self.vx} :: FAZE 2")
                        self.test_phase = 2
                        self.ph_1_end_time = current_time

                case 2:
                    # Faze 2: Pokud jsme zhruba dosahli cilove rychlosti, zacneme pomalu zvetsovat uhel zatoceni
                    time_from_ph_1 = current_time - self.ph_1_end_time

                    rpm_ff = self.vx2rpm(self.target_speed_m_s)
                    rpm_fb = self.speed2rpm_controller.step(self.target_speed_m_s - self.vx, self.timer_period) # type: ignore
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req = steer_rate_rad_s * time_from_ph_1
                    self.get_logger().info(f"Zatacim: {steering_angle_req:.2f} deg")

            # Logovani pozadavku
            self.log_commands(rpm_req, steering_angle_req)

            # Publikovani pozadavku
            self.publish_data(rpm_req, steering_angle_req)
            
            # Logovani do CSV
            self.log_data(rpm_req, steering_angle_req)

        return True
    
        # ROS2 logovani
        # otevrit v druhem terminalu:
        # ros2 bag record -o muj_rampsteer_bag /state/odometry /servo/front/angle_req /powertrain/rl/rpm_req /powertrain/rr/rpm_req

class RampSpeedTest(TestBluePrint):
    def __init__(self, node_name: str = 'ramp_speed_test', start_speed_m_s: float = 10.0, target_steer_deg: float = 5.0, acceleration_m_ss: float = 1.0, test_length: float = 20.0, timer_period: float = 0.1):
        super().__init__(node_name=node_name,
                         control_loop=self.control_loop,
                         timer_period=timer_period
                         )
        
        self.start_speed = start_speed_m_s
        self.target_steer_deg = target_steer_deg
        self.acceleration = acceleration_m_ss
        self.timer_period = timer_period          # seconds
        self.test_length = test_length      # seconds

        self.speed2rpm_controller = PIDController(
            gains=PIDGains(kp=70, ki=150, kd=0),
            output_limit=(-200, 200),
            integrator_limit=(-200, 200),
        )

        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.test_phase = 1
        self.ph_1_end_time = 0.0
        self.ph_2_end_time = 0.0
        self.ph_3_end_time = 0.0

    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed_time = current_time - self.start_time
        rpm_req = 0.0
        steering_angle_req = 0.0


        if elapsed_time >= self.test_length:  # Test length in seconds
            self.get_logger().info("Test length exceeded. Stopping test.")
            return False    

        if self.all_data_available():
            match self.test_phase:

                case 1:
                    # Faze 1: Zrychleni na cilovou rychlost 
                    tar_sp = self.start_speed
                    rpm_ff = self.vx2rpm(tar_sp)
                    rpm_fb = self.speed2rpm_controller.step(tar_sp - self.vx, self.timer_period) # type: ignore
                    self.get_logger().info(f"Error 1: {(tar_sp - self.vx):.2f} m/s -- rpm fb: {(rpm_fb):.2f}")
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req = 0.0

                    if self.vx >= self.start_speed * 0.95: # type: ignore
                        self.test_phase = 2
                        self.ph_1_end_time = current_time

                case 2:
                    # Faze 2: Drzet cilovou rychlost a zatocit na pozadovany uhel
                    phase_2_duration = 1 # jak rychle ma auto skubnout koly
                    tar_sp = self.start_speed
                    rpm_ff = self.vx2rpm(tar_sp)
                    rpm_fb = self.speed2rpm_controller.step(self.start_speed - self.vx, self.timer_period) # type: ignore
                    self.get_logger().info(f"Error 2: {(self.start_speed - self.vx):.2f} m/s -- rpm fb: {(rpm_fb):.2f}")
                    rpm_req = rpm_ff + rpm_fb
                    time_from_ph_1 = current_time - self.ph_1_end_time
                    steering_angle_req = self.deg2rad(self.target_steer_deg) * (time_from_ph_1 / phase_2_duration)

                    if time_from_ph_1 > phase_2_duration:
                        self.test_phase = 3
                        self.ph_2_end_time = current_time

                case 3:
                    # Faze 3: Zacit pomale zrychlovat pri zachovani steering angle
                    time_from_ph_2 = current_time - self.ph_2_end_time
                    tar_sp = self.start_speed + self.acceleration * time_from_ph_2
                    rpm_ff = self.vx2rpm(tar_sp)
                    rpm_fb = self.speed2rpm_controller.step(tar_sp - self.vx, self.timer_period) # type: ignore
                    self.get_logger().info(f"Error 3: {(tar_sp - self.vx):.2f} m/s -- rpm fb: {(rpm_fb):.2f}")
                    rpm_req = rpm_ff + rpm_fb
                    steering_angle_req = self.deg2rad(self.target_steer_deg)

            # Logovani pozadavku
            self.log_commands(rpm_req, steering_angle_req)

            # Publikovani pozadavku
            self.publish_data(rpm_req, steering_angle_req)
            
            # Logovani do CSV
            self.log_data(rpm_req, steering_angle_req)

        return True

class CombinedStressTest:
    """Orchestrátor pro zmapování kombinovaného skluzu.
    Spouští sekvenčně sérii RampSpeedTestů s rostoucím úhlem natočení.
    """
    def __init__(self, 
                 start_speed_m_s: float = 3.0, 
                 max_steer_deg: float = 20.0, 
                 num_branches: int = 4, 
                 acceleration_m_ss: float = 5.0, 
                 branch_test_length: float = 12.0, 
                 timer_period: float = 0.1):
        
        self.start_speed_m_s = start_speed_m_s
        self.acceleration_m_ss = acceleration_m_ss
        self.branch_test_length = branch_test_length
        self.timer_period = timer_period
        
        # Vygenerování úhlů, např. pro max=20 a num=4 vygeneruje [5.0, 10.0, 15.0, 20.0]
        self.steer_angles = np.linspace(
            max_steer_deg / num_branches, 
            max_steer_deg, 
            num_branches
        )

    def run(self):
        for i, steer_deg in enumerate(self.steer_angles):
            print(f"\n=======================================================")
            print(f"--- Spouštím RampSpeedTest {i+1}/{len(self.steer_angles)} (Úhel: {steer_deg:.1f}°) ---")
            print(f"=======================================================\n")
            
            # 1. Inicializace nového ROS uzlu pro konkrétní větev (vytvoří si vlastní CSV soubor)
            name = str(f"ramp_speed_test_{int(steer_deg)}")
            node = RampSpeedTest(
                node_name=name,
                start_speed_m_s=self.start_speed_m_s,
                target_steer_deg=steer_deg,
                acceleration_m_ss=self.acceleration_m_ss,
                test_length=self.branch_test_length,
                timer_period=self.timer_period
            )
            
            # 2. Běh smyčky pro daný uzel
            try:
                while rclpy.ok() and not node.should_stop:
                    rclpy.spin_once(node, timeout_sec=self.timer_period)
            except KeyboardInterrupt:
                if node is not None:
                    node.destroy_blueprint_node()
                return
            finally:
                if node is not None:
                    try:
                        node.destroy_blueprint_node()
                    except Exception:
                        pass
            
            # 3. Pauza pro reset dynamiky auta (zastavení) a logů
            if i < len(self.steer_angles) - 1:
                input(f"Test {i+1} dokončen. Srovnejte auto a stiskněte [ENTER] pro další test...")

        print("\n*** Všechny větve CombinedStressTestu úspěšně dokončeny! ***")

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    
    # multiline (un)comment -> Ctrl + /

    """ INIT TEST SETUP """
    """
    - overit, ze vse funguje, ze jsou vsechna data logovana a nic nepada
    """
    # test_length = 20.0  # seconds
    # timer_period = 0.05  # 20 Hz
    # node = InitTest(
    #     node_name="first_test_eva",
    #     test_length=test_length,
    #     timer_period=timer_period,
    # )

    """" SINE WAVE TEST SETUP """
    # target_speed_m_s = 1.0
    # sine_freq = 1.0
    # sine_ampl = 1.0
    # test_length = 10.0
    # timer_period = 0.01

    # node = SineWaveTest(
    #     target_speed_m_s=target_speed_m_s,
    #     sine_freq=sine_freq,
    #     sine_ampl=sine_ampl,
    #     test_length=test_length,
    #     timer_period=timer_period,
    # )

    """ RAMP STEER TEST SETUP """
    """
    - slip ratio = 0 (vx se v prubehu testu nemeni) 
    - -> identifikace prubehu pricne sily v zavislosti na alfa
    vysledek:
    - ciste pricna dynamika pneumatiky pro predek i zadek
    """
    # target_speed_m_s = 10.0  # m/s
    # steering_rate = 1.0  # degrees
    # test_length = 20.0  # seconds
    # timer_period = 0.1  # 10 Hz

    # node = RampSteerTest(
    #     target_speed_m_s=target_speed_m_s,
    #     steering_angle_rate_deg_s=steering_rate,
    #     test_length=test_length,
    #     timer_period=timer_period,
    # )

    """ RAMP SPEED TEST SETUP """
    """
    - pro nejake nami vybane definovane alfa zacit zrychlovat (ale dostatecne pomalu pro male slip ratio)
    - taky zmeri pouze pricnou dynamiku vozidla 
    - casem se to auto urve 
    """
    start_speed_m_s = 1.0 
    target_steer_deg = 20.0
    acceleration_m_ss = 0.3  
    test_length = 20.0  # seconds
    timer_period = 0.01  # 100 Hz
    node = RampSpeedTest(
        node_name=f"rst_{int(target_steer_deg)}_{int(acceleration_m_ss)}",
        start_speed_m_s=start_speed_m_s,
        target_steer_deg=target_steer_deg,
        acceleration_m_ss=acceleration_m_ss,
        test_length=test_length,
        timer_period=timer_period
    )


    """ NORMALNI SPUSTENI TESTU """
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

    """ SPUSTENI SERIE TESTU PRO COMBINED SLIP """
    """
    - snaha identifikovat chovani pri combined slip
    - zrychleni uz musi byt dostatecne velke, aby se slip ratio projevilo
    - pokus zmapovat sirsi okoli side slip angle
    -- principielne nebudu ani znat ciste podelnou dynamiku vozidla z cele serie testu
    """
    # # Vytvoření orchestrátoru
    # orchestrator = CombinedStressTest(
    #     start_speed_m_s=3.0,         # Rychlost pro zachycení čistého gripu
    #     max_steer_deg=20.0,          # Maximální úhel, kam až chceme mapovat (např. 20°)
    #     num_branches=4,              # Počet větví (vygeneruje testy pro 5°, 10°, 15°, 20°)
    #     acceleration_m_ss=3.0,       # Pomalé přidávání plynu v driftu (m/s^2)
    #     branch_test_length=12.0,     # Délka trvání JEDNOHO testu (v sekundách)
    #     timer_period=0.01
    # )

    # try:
    #     # Tímto se spustí celá série testů. Blokuje to, dokud se všechny nedokončí.
    #     orchestrator.run()
    # finally:
    #     rclpy.shutdown()

if __name__ == '__main__':
    main()
