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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.processor import make_default_processors
from lerobot.robots.ur_follower import URFollower, URFollowerConfig
from lerobot.robots.ur_follower.ur_follower import EE_POSE_KEYS, GRIPPER_OPEN, JOINT_NAMES

_MODULE = "lerobot.robots.ur_follower.ur_follower"


@pytest.fixture
def follower():
    control = MagicMock(name="RTDEControl")
    receive = MagicMock(name="RTDEReceive")
    io = MagicMock(name="RTDEIO")
    receive.getActualQ.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    receive.getActualTCPPose.return_value = [0.4, 0.1, 0.2, 0.0, 3.14, 0.0]
    receive.getDigitalOutState.return_value = True

    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.RTDEControlInterface", return_value=control),
        patch(f"{_MODULE}.RTDEReceiveInterface", return_value=receive),
        patch(f"{_MODULE}.RTDEIOInterface", return_value=io),
    ):
        robot = URFollower(URFollowerConfig(ip="192.168.0.10", max_relative_target=0.05))
        robot.connect()
        yield robot, control, receive, io
        if robot.is_connected:
            robot.disconnect()


def test_features_are_stable_before_connect():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        robot = URFollower(URFollowerConfig(ip="192.168.0.10"))

    assert set(robot.action_features) == {*EE_POSE_KEYS, GRIPPER_OPEN}
    assert {f"{joint}.pos" for joint in JOINT_NAMES}.issubset(robot.observation_features)
    assert set(EE_POSE_KEYS).issubset(robot.observation_features)
    assert GRIPPER_OPEN in robot.observation_features


def test_connect_disconnect(follower):
    robot, control, _receive, _io = follower
    assert robot.is_connected

    robot.disconnect()

    control.servoStop.assert_called_once()
    control.stopScript.assert_called_once()
    assert not robot.is_connected


def test_get_observation_returns_joints_pose_and_gripper(follower):
    robot, _control, _receive, _io = follower

    obs = robot.get_observation()

    assert obs["shoulder_pan.pos"] == 0.1
    assert obs["wrist_3.pos"] == 0.6
    assert obs["ee.x"] == 0.4
    assert obs["ee.ry"] == 3.14
    assert obs[GRIPPER_OPEN] == 1.0


def test_send_action_uses_servo_l_and_digital_out(follower):
    robot, control, _receive, io = follower
    action = {
        "ee.x": 0.6,
        "ee.y": 0.1,
        "ee.z": 0.2,
        "ee.rx": 0.0,
        "ee.ry": 3.14,
        "ee.rz": 0.0,
        GRIPPER_OPEN: 1.0,
    }

    returned = robot.send_action(action)

    # ee.x is clipped from 0.6 to current 0.4 + max_relative_target 0.05.
    assert returned["ee.x"] == pytest.approx(0.45)
    assert returned[GRIPPER_OPEN] == 1.0
    control.servoL.assert_called_once()
    sent_pose = control.servoL.call_args.args[0]
    assert sent_pose[0] == pytest.approx(0.45)
    io.setStandardDigitalOut.assert_called_once_with(0, True)


def test_send_action_closes_gripper_with_inverse_state(follower):
    robot, _control, _receive, io = follower

    returned = robot.send_action({GRIPPER_OPEN: 0.0})

    assert returned[GRIPPER_OPEN] == 0.0
    io.setStandardDigitalOut.assert_called_once_with(0, False)


def test_send_action_clips_to_workspace():
    control = MagicMock(name="RTDEControl")
    receive = MagicMock(name="RTDEReceive")
    io = MagicMock(name="RTDEIO")
    receive.getActualTCPPose.return_value = [0.4, 0.1, 0.2, 0.0, 3.14, 0.0]

    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.RTDEControlInterface", return_value=control),
        patch(f"{_MODULE}.RTDEReceiveInterface", return_value=receive),
        patch(f"{_MODULE}.RTDEIOInterface", return_value=io),
    ):
        robot = URFollower(
            URFollowerConfig(
                ip="192.168.0.10",
                max_relative_target=None,
                workspace_bounds={"ee.x": (0.0, 0.5)},
            )
        )
        robot.connect()
        returned = robot.send_action({"ee.x": 0.8})

    assert returned["ee.x"] == 0.5
    assert control.servoL.call_args.args[0][0] == 0.5


def test_default_processors_map_gamepad_delta_to_ur_pose(follower):
    robot, _control, _receive, _io = follower
    teleop = MagicMock(name="GamepadTeleop")
    teleop.name = "gamepad"

    teleop_action_processor, _robot_action_processor, _obs_processor = make_default_processors(
        robot=robot, teleop=teleop
    )
    observation = {
        "ee.x": 0.4,
        "ee.y": 0.1,
        "ee.z": 0.2,
        "ee.rx": 0.0,
        "ee.ry": 3.14,
        "ee.rz": 0.0,
        GRIPPER_OPEN: 0.0,
    }

    action = teleop_action_processor(({"delta_x": 1.0, "delta_y": 0.0, "delta_z": -1.0, "gripper": 2.0}, observation))

    assert action["ee.x"] == pytest.approx(0.41)
    assert action["ee.y"] == pytest.approx(0.1)
    assert action["ee.z"] == pytest.approx(0.19)
    assert action["ee.ry"] == pytest.approx(3.14)
    assert action[GRIPPER_OPEN] == 1.0
