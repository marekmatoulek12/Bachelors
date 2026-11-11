"""Tests for keyboard-controller terminal input and command publishing."""

import os
import pty
import tty

from car_tests_pkg.Keyboard_Controller import KeyboardController


def test_reads_all_available_terminal_keys_without_buffering():
    """Read multiple keypresses in one poll and leave no buffered leftovers."""
    master_fd, slave_fd = pty.openpty()
    try:
        tty.setraw(slave_fd)
        controller = KeyboardController.__new__(KeyboardController)
        controller.stdin_fd = slave_fd

        os.write(master_fd, b'wa')

        assert controller.get_keys_non_blocking() == ['w', 'a']
        assert controller.get_keys_non_blocking() == []
    finally:
        os.close(master_fd)
        os.close(slave_fd)


def test_combines_steering_and_drive_keys_across_input_cycles(monkeypatch):
    """Keep simultaneous drive and steering commands from separate reads."""
    controller = KeyboardController.__new__(KeyboardController)
    controller.forward_speed = 10.0
    controller.boost_multiplier = 2.0
    controller.steering_angle = 0.5
    controller.speed_command = None
    controller.speed_command_time = 0.0
    controller.steering_command = None
    controller.steering_command_time = 0.0
    controller.KEY_HOLD_TIMEOUT = 0.2
    commands = []
    controller.log_commands = lambda rpm, steer: commands.append((rpm, steer))
    controller.publish_data = lambda: None
    key_cycles = iter((['w'], ['a'], []))
    times = iter((1.0, 1.05, 1.1))
    controller.get_keys_non_blocking = lambda: next(key_cycles)
    monkeypatch.setattr(
        'car_tests_pkg.Keyboard_Controller.time.monotonic',
        lambda: next(times),
    )

    assert controller.control_loop()
    assert controller.control_loop()
    assert controller.control_loop()
    assert commands == [(10.0, 0.0), (10.0, 0.5), (10.0, 0.5)]


def test_commands_expire_after_keys_are_released(monkeypatch):
    """Stop each axis when its most recent terminal key event expires."""
    controller = KeyboardController.__new__(KeyboardController)
    controller.forward_speed = 10.0
    controller.boost_multiplier = 2.0
    controller.steering_angle = 0.5
    controller.speed_command = None
    controller.speed_command_time = 0.0
    controller.steering_command = None
    controller.steering_command_time = 0.0
    controller.KEY_HOLD_TIMEOUT = 0.2
    commands = []
    controller.log_commands = lambda rpm, steer: commands.append((rpm, steer))
    controller.publish_data = lambda: None
    key_cycles = iter((['w', 'a'], []))
    times = iter((1.0, 1.21))
    controller.get_keys_non_blocking = lambda: next(key_cycles)
    monkeypatch.setattr(
        'car_tests_pkg.Keyboard_Controller.time.monotonic',
        lambda: next(times),
    )

    assert controller.control_loop()
    assert controller.control_loop()
    assert commands == [(10.0, 0.5), (0.0, 0.0)]


def test_space_immediately_stops_both_axes(monkeypatch):
    """Clear active drive and steering requests when space is pressed."""
    controller = KeyboardController.__new__(KeyboardController)
    controller.forward_speed = 10.0
    controller.boost_multiplier = 2.0
    controller.steering_angle = 0.5
    controller.speed_command = None
    controller.speed_command_time = 0.0
    controller.steering_command = None
    controller.steering_command_time = 0.0
    controller.KEY_HOLD_TIMEOUT = 0.2
    commands = []
    controller.log_commands = lambda rpm, steer: commands.append((rpm, steer))
    controller.publish_data = lambda: None
    controller.get_keys_non_blocking = lambda: ['w', 'a', ' ']
    monkeypatch.setattr(
        'car_tests_pkg.Keyboard_Controller.time.monotonic',
        lambda: 1.0,
    )

    assert controller.control_loop()
    assert commands == [(0.0, 0.0)]


def test_zero_steering_request_publishes_zero_radians():
    """Publish a neutral steering angle as zero radians."""

    class Publisher:
        """Capture published messages for assertions."""

        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    controller = KeyboardController.__new__(KeyboardController)
    controller.steer_rad_req = 0.0
    controller.rpm_req = 0.0
    controller.pub_steer = Publisher()
    controller.pub_rpm_rl = Publisher()
    controller.pub_rpm_rr = Publisher()

    controller.publish_data()

    assert controller.pub_steer.messages[0].data == 0.0
    assert controller.pub_rpm_rl.messages[0].data == 0.0
    assert controller.pub_rpm_rr.messages[0].data == 0.0
