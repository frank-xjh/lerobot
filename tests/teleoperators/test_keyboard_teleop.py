#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import patch

from lerobot.teleoperators.keyboard import teleop_keyboard as keyboard_module
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop

_MODULE = "lerobot.teleoperators.keyboard.teleop_keyboard"


class _Key:
    up = object()
    down = object()
    left = object()
    right = object()
    shift = object()
    shift_r = object()
    ctrl_l = object()
    ctrl_r = object()
    esc = object()


def _make_keyboard_ee(monkeypatch):
    monkeypatch.setattr(keyboard_module, "keyboard", SimpleNamespace(Key=_Key))
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        return KeyboardEndEffectorTeleop(KeyboardEndEffectorTeleopConfig())


def _get_action_without_connection_check(teleop):
    return KeyboardEndEffectorTeleop.get_action.__wrapped__(teleop)


def test_keyboard_ee_ignores_stale_released_opposite_keys(monkeypatch):
    teleop = _make_keyboard_ee(monkeypatch)
    teleop.current_pressed = {
        _Key.left: True,
        _Key.right: False,
        _Key.up: True,
        _Key.down: False,
        _Key.shift: True,
        _Key.shift_r: False,
    }

    action = _get_action_without_connection_check(teleop)

    assert action["delta_x"] == 1.0
    assert action["delta_y"] == -1.0
    assert action["delta_z"] == -1.0


def test_keyboard_ee_removes_released_keys_before_next_action(monkeypatch):
    teleop = _make_keyboard_ee(monkeypatch)
    teleop.event_queue.put((_Key.left, True))
    assert _get_action_without_connection_check(teleop)["delta_x"] == 1.0

    teleop.event_queue.put((_Key.left, False))
    teleop.event_queue.put((_Key.right, True))
    action = _get_action_without_connection_check(teleop)

    assert _Key.left not in teleop.current_pressed
    assert action["delta_x"] == -1.0
