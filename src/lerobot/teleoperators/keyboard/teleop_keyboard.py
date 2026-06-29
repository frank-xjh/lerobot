#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import select
import sys
import time
from queue import Queue
from typing import Any

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - POSIX-only terminal teleop.
    termios = None
    tty = None

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardTeleopConfig,
    TerminalKeyboardEndEffectorTeleopConfig,
)

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info("Could not import pynput keyboard backend: %s", e)


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        if PYNPUT_AVAILABLE and pynput_can_capture():
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.warning(
                "Keyboard teleoperation is unavailable in this environment. pynput can only "
                "capture key events on an X11 session (Linux), a Windows desktop, or macOS with "
                "Accessibility / Input Monitoring granted - not on Wayland or headless machines. "
                "This keyboard teleoperator will produce no actions; use an X11 session, a "
                "gamepad, or a leader-arm teleoperator instead."
            )
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                self.current_pressed.pop(key_char, None)
            self._on_key_state_change(key_char, is_pressed)

    def _on_key_state_change(self, key, is_pressed: bool) -> None:
        pass

    def configure(self):
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()
        self._axis_direction = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._gripper_action = 1.0

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        active_keys = {key for key, val in self.current_pressed.items() if val}

        for key in active_keys:
            if key not in {
                keyboard.Key.up,
                keyboard.Key.down,
                keyboard.Key.left,
                keyboard.Key.right,
                keyboard.Key.shift,
                keyboard.Key.shift_r,
                keyboard.Key.ctrl_r,
                keyboard.Key.ctrl_l,
            }:
                # This records key presses that are not part of the motion action,
                # such as episode success/rerecord/quit events.
                self.misc_keys_queue.put(key)

        action_dict = {
            "delta_x": self._axis_direction["x"],
            "delta_y": self._axis_direction["y"],
            "delta_z": self._axis_direction["z"],
        }

        if self.config.use_gripper:
            action_dict["gripper"] = self._gripper_action

        return action_dict

    def _on_key_state_change(self, key, is_pressed: bool) -> None:
        direction_by_key = {
            keyboard.Key.up: ("y", -1.0),
            keyboard.Key.down: ("y", 1.0),
            keyboard.Key.left: ("x", 1.0),
            keyboard.Key.right: ("x", -1.0),
            keyboard.Key.shift: ("z", -1.0),
            keyboard.Key.shift_r: ("z", 1.0),
        }
        if key in direction_by_key:
            axis, direction = direction_by_key[key]
            if is_pressed:
                self._axis_direction[axis] = direction
            elif self._axis_direction[axis] == direction:
                self._axis_direction[axis] = 0.0
            return

        gripper_by_key = {
            keyboard.Key.ctrl_l: 0.0,
            keyboard.Key.ctrl_r: 2.0,
        }
        if key in gripper_by_key:
            action = gripper_by_key[key]
            if is_pressed:
                self._gripper_action = action
            elif self._gripper_action == action:
                self._gripper_action = 1.0

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right,
            keyboard.Key.shift,
            keyboard.Key.shift_r,
            keyboard.Key.ctrl_r,
            keyboard.Key.ctrl_l,
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        self.current_pressed.clear()

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class TerminalKeyboardEndEffectorTeleop(Teleoperator):
    """End-effector teleoperation from an interactive terminal.

    This is intended for Linux SSH/headless sessions where pynput cannot capture
    global keyboard events.
    """

    config_class = TerminalKeyboardEndEffectorTeleopConfig
    name = "terminal_keyboard_ee"

    def __init__(self, config: TerminalKeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self._old_terminal_settings = None
        self._axis_direction = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._axis_updated_at = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._gripper_action = 1.0
        self._gripper_updated_at = 0.0

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        return {
            "dtype": "float32",
            "shape": (3,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._old_terminal_settings is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self) -> None:
        if termios is None or tty is None:
            raise ConnectionError("terminal_keyboard_ee is only supported on POSIX terminals.")
        if not sys.stdin.isatty():
            raise ConnectionError("terminal_keyboard_ee requires an interactive terminal.")
        self._old_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        print(
            "Terminal keyboard EE controls: arrows or WASD move x/y, U/J move z, "
            "O/C open/close gripper, Space stops motion."
        )

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.perf_counter()
        for key in self._read_keys():
            self._handle_key(key, now)

        for axis, updated_at in self._axis_updated_at.items():
            if now - updated_at > self.config.command_timeout_s:
                self._axis_direction[axis] = 0.0

        if now - self._gripper_updated_at > self.config.command_timeout_s:
            self._gripper_action = 1.0

        action_dict = {
            "delta_x": self._axis_direction["x"],
            "delta_y": self._axis_direction["y"],
            "delta_z": self._axis_direction["z"],
        }
        if self.config.use_gripper:
            action_dict["gripper"] = self._gripper_action
        return action_dict

    def _read_keys(self) -> list[str]:
        chars = []
        while select.select([sys.stdin], [], [], 0)[0]:
            chars.append(sys.stdin.read(1))

        keys = []
        i = 0
        while i < len(chars):
            if chars[i] == "\x1b" and i + 2 < len(chars) and chars[i + 1] == "[":
                arrow_key = {
                    "A": "up",
                    "B": "down",
                    "C": "right",
                    "D": "left",
                }.get(chars[i + 2])
                if arrow_key is not None:
                    keys.append(arrow_key)
                    i += 3
                    continue
            keys.append(chars[i].lower())
            i += 1
        return keys

    def _handle_key(self, key: str, now: float) -> None:
        if key in {" ", "x"}:
            self._axis_direction = {"x": 0.0, "y": 0.0, "z": 0.0}
            self._gripper_action = 1.0
            return

        direction_by_key = {
            "left": ("x", 1.0),
            "a": ("x", 1.0),
            "right": ("x", -1.0),
            "d": ("x", -1.0),
            "up": ("y", -1.0),
            "w": ("y", -1.0),
            "down": ("y", 1.0),
            "s": ("y", 1.0),
            "u": ("z", 1.0),
            "j": ("z", -1.0),
        }
        if key in direction_by_key:
            axis, direction = direction_by_key[key]
            self._axis_direction[axis] = direction
            self._axis_updated_at[axis] = now
            return

        gripper_by_key = {"o": 2.0, "c": 0.0}
        if key in gripper_by_key:
            self._gripper_action = gripper_by_key[key]
            self._gripper_updated_at = now

    def get_teleop_events(self) -> dict[str, Any]:
        return {
            TeleopEvents.IS_INTERVENTION: any(value != 0.0 for value in self._axis_direction.values()),
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_terminal_settings)
        self._old_terminal_settings = None


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    Keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Provides intuitive WASD-style controls for driving a mobile robot:
    - Linear movement (forward/backward)
    - Angular movement (turning/rotation)
    - Speed adjustment
    - Emergency stop

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Turn left (with forward motion)
            - D: Turn right (with forward motion)
            - Q: Rotate left in place
            - E: Rotate right in place
            - X: Emergency stop

        Speed Control:
            - +/=: Increase speed
            - -: Decrease speed

        System:
            - ESC: Disconnect teleoperator

    Attributes:
        config: Teleoperator configuration
        current_linear_speed: Current linear velocity magnitude
        current_angular_speed: Current angular velocity magnitude

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """Return action format for rover (linear and angular velocities)."""
        return {
            "linear_velocity": float,
            "angular_velocity": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Rover teleop doesn't require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue without clearing held keys"""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # Only remove key if it's being released
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get the current action based on pressed keys.

        Returns:
            RobotAction with 'linear_velocity' and 'angular_velocity' keys.
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        # Check which keys are currently pressed (not released)
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        # Linear movement (W/S) - these take priority
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # Turning (A/D/Q/E)
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # Rotate in place
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # Rotate in place

        # Stop (X) - overrides everything
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # Speed adjustment
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
