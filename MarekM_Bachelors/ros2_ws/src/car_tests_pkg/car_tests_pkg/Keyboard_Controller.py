import signal
import sys
import select
import termios
import tty
import rclpy
from rclpy.signals import SignalHandlerOptions
from .Blueprints import SimpleConnectionBP

class KeyboardController(SimpleConnectionBP):
    def __init__(self, timer_period: float, node_name: str = "Keyboard Controller"):
        super().__init__(node_name=node_name,
                         control_loop=self.control_loop,
                         timer_period=timer_period,
                         )

        FORWARD_SPEED = 1 # [m/s]
        BOOST_MULTIPLIER = 2.0 # [-]
        STEERING_ANGLE = 20.0 # [deg]

        self.forward_speed = self.vx2rpm(FORWARD_SPEED)
        self.boost_multiplier = BOOST_MULTIPLIER
        self.steering_angle = self.deg2rad(STEERING_ANGLE)

    def get_keys_non_blocking(self):
        """
        Vyčte všechny aktuálně stisknuté klávesy v bufferu.
        Vrací seznam znaků (např. ['w', 'a']).
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        keys = []
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                # 0.0 timeout zaručí, že to nečeká - jakmile je buffer prázdný, smyčka skončí
                rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                if rlist:
                    keys.append(sys.stdin.read(1))
                else:
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return keys

    def control_loop(self):
        # Lokální proměnné (nulují se každou iteraci - auto zastaví, jakmile klávesy pustíte)
        rpm_req = 0.0
        steer_rad = 0.0
        
        keys = self.get_keys_non_blocking()
        
        # Zpracování všech znaků v bufferu
        for key in keys:
            # 1. Nezávislý blok pro plyn
            if key == 'w':
                rpm_req = self.forward_speed
            elif key == 'W':
                rpm_req = self.forward_speed * self.boost_multiplier
            elif key == 's':
                rpm_req = -(self.forward_speed)
            elif key == 'S':
                rpm_req = -(self.forward_speed * self.boost_multiplier)
                
            # 2. Nezávislý blok pro zatáčení
            if key in ['a', 'A']:
                steer_rad = self.steering_angle
            elif key in ['d', 'D']:
                steer_rad = -(self.steering_angle)
                
            # 3. Speciální příkazy (přepisují vše ostatní)
            if key == ' ':
                rpm_req = 0.0
                steer_rad = 0.0
            elif key in ['q', 'Q', '\x03']: # \x03 je Ctrl+C
                self.log_commands(0.0, 0.0)
                self.publish_data()
                return False


        # Logování a publikace pouze pomocí lokálních proměnných
        self.log_commands(rpm_req, steer_rad)
        self.publish_data()

        return True

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)

    timer_period = 0.05 # Rychlejší čtení pro plynulejší ovládání klávesnicí
    node = KeyboardController(timer_period=timer_period)

    print("\n--- KEYBOARD CONTROLLER ACTIVATED ---")
    print("W / S : Zrychlit vpřed / vzad (250 RPM)")
    print("A / D : Zatáčet vlevo / vpravo (+-20 deg)")
    print("SHIFT + W / S : Zrychlit vpřed / vzad (500 RPM)")
    print("SPACE : Okamžitě zastavit (0 RPM, 0 deg)")
    print("Q nebo Ctrl+C : Ukončit skript\n")

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

if __name__ == '__main__':
    main()