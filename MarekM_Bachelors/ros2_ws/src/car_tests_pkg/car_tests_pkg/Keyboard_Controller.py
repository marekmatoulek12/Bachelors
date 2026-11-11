"""Terminal keyboard controller for the CogniCar."""

import os
import select
import signal
import sys
import termios
import time
import tty

import rclpy
from rclpy.signals import SignalHandlerOptions

from .Blueprints import SimpleConnectionBP


class KeyboardController(SimpleConnectionBP):
    """Drive the car from keys pressed in an interactive terminal."""

    # Zvětšeno na 0.4s, aby výkyvy v opakování kláves od OS nezpůsobovaly výpadky
    KEY_HOLD_TIMEOUT = 0.4

    def __init__(
        self, timer_period: float, node_name: str = 'Keyboard_Controller'
    ):
        """Create the controller and configure its keyboard commands."""
        super().__init__(node_name=node_name,
                         control_loop=self.control_loop,
                         timer_period=timer_period,
                         )
        self.stdin_fd = sys.stdin.fileno()

        FORWARD_SPEED = 1  # [m/s]
        BOOST_MULTIPLIER = 2.0  # [-]
        STEERING_ANGLE = 20.0  # [deg]

        self.forward_speed = self.vx2rpm(FORWARD_SPEED)
        self.boost_multiplier = BOOST_MULTIPLIER
        self.steering_angle = self.deg2rad(STEERING_ANGLE)
        
        # Oddělené uchování stavu a času pro plyn a zatáčení
        self.current_speed = 0.0
        self.speed_command_time = 0.0
        
        self.current_steering = 0.0
        self.steering_command_time = 0.0

    def get_keys_non_blocking(self):
        """Read all available keypresses without blocking the ROS loop."""
        ready, _, _ = select.select([self.stdin_fd], [], [], 0.0)
        if not ready:
            return []
        keys = os.read(self.stdin_fd, 1024).decode('ascii', errors='ignore')
        return list(keys)

    def control_loop(self):
        """Publish motion commands for the keys pressed during this cycle."""
        keys = self.get_keys_non_blocking()
        current_time = time.monotonic()

        # Zpracování všech přijatých kláves v aktuálním tiku
        for key in keys:
            if key == 'w':
                self.current_speed = self.forward_speed
                self.speed_command_time = current_time
            elif key == 'W':
                self.current_speed = self.forward_speed * self.boost_multiplier
                self.speed_command_time = current_time
            elif key == 's':
                self.current_speed = -self.forward_speed
                self.speed_command_time = current_time
            elif key == 'S':
                self.current_speed = -self.forward_speed * self.boost_multiplier
                self.speed_command_time = current_time

            if key in ['a', 'A']:
                self.current_steering = self.steering_angle
                self.steering_command_time = current_time
            elif key in ['d', 'D']:
                self.current_steering = -self.steering_angle
                self.steering_command_time = current_time

            if key == ' ':
                self.current_speed = 0.0
                self.current_steering = 0.0
                self.speed_command_time = 0.0
                self.steering_command_time = 0.0
            elif key in ['q', 'Q', '\x03']:
                self.current_speed = 0.0
                self.current_steering = 0.0
                self.log_commands(0.0, 0.0)
                self.publish_data()
                return False

        # Vyhodnocení platnosti plynu (zda byl W/S stisknut v rámci timeoutu)
        if current_time - self.speed_command_time <= self.KEY_HOLD_TIMEOUT:
            rpm_req = self.current_speed
        else:
            rpm_req = 0.0

        # Vyhodnocení platnosti zatáčení (zda byl A/D stisknut v rámci timeoutu)
        if current_time - self.steering_command_time <= self.KEY_HOLD_TIMEOUT:
            steer_rad = self.current_steering
        else:
            steer_rad = 0.0

        # Odeslání obou příkazů současně
        self.log_commands(rpm_req, steer_rad)
        self.publish_data()

        return True


def main(args=None):
    """Run the keyboard controller until the user quits."""
    stdin_fd = sys.stdin.fileno()
    if not os.isatty(stdin_fd):
        raise RuntimeError(
            'Run keyboard_controller from an interactive terminal.'
        )

    original_terminal_settings = termios.tcgetattr(stdin_fd)
    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    node = None
    ros_initialized = False
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        tty.setraw(stdin_fd)
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        ros_initialized = True

        timer_period = 0.05
        node = KeyboardController(timer_period=timer_period)

        print('\n--- KEYBOARD CONTROLLER ACTIVATED ---')
        print('Hold W / S to drive forward / backward (SHIFT for faster).')
        print('Hold A / D to steer left / right (+/-20 deg).')
        print('SPACE stops; released-key commands expire after 0.4 seconds.')
        print('Q or Ctrl+C exits.\n')

        while rclpy.ok() and not node.should_stop and not stop_requested:
            rclpy.spin_once(node, timeout_sec=timer_period)
    finally:
        try:
            if node is not None:
                node.destroy_blueprint_node()
        finally:
            try:
                if ros_initialized and rclpy.ok():
                    rclpy.shutdown()
            finally:
                try:
                    termios.tcsetattr(
                        stdin_fd, termios.TCSADRAIN, original_terminal_settings
                    )
                finally:
                    signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == '__main__':
    main()