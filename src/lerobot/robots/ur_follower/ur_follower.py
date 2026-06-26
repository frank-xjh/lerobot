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

import logging
import time
from functools import cached_property
from typing import TYPE_CHECKING

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _ur_rtde_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_ur_follower import URFollowerConfig

if TYPE_CHECKING or _ur_rtde_available:
    from rtde_control import RTDEControlInterface
    from rtde_io import RTDEIOInterface
    from rtde_receive import RTDEReceiveInterface
else:
    RTDEControlInterface = None
    RTDEIOInterface = None
    RTDEReceiveInterface = None

logger = logging.getLogger(__name__)

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
EE_POSE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz")
GRIPPER_OPEN = "gripper.open"


class URFollower(Robot):
    """Universal Robots follower controlled through ur_rtde.

    Actions are expressed as TCP pose targets plus a digital-output gripper
    command. Joint positions are still observed to preserve the robot
    configuration in recorded datasets.
    """

    config_class = URFollowerConfig
    name = "ur_follower"

    def __init__(self, config: URFollowerConfig):
        require_package("ur-rtde", extra="ur", import_name="rtde_control")
        super().__init__(config)
        self.config = config
        self.rtde_control: RTDEControlInterface | None = None
        self.rtde_receive: RTDEReceiveInterface | None = None
        self.rtde_io: RTDEIOInterface | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _joint_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in JOINT_NAMES}

    @property
    def _ee_pose_ft(self) -> dict[str, type]:
        return dict.fromkeys(EE_POSE_KEYS, float)

    @property
    def _gripper_ft(self) -> dict[str, type]:
        return {GRIPPER_OPEN: float}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joint_ft, **self._ee_pose_ft, **self._gripper_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._ee_pose_ft, **self._gripper_ft}

    def default_teleop_action_processor_steps(self, teleop=None) -> list:
        if getattr(teleop, "name", None) in {"gamepad", "keyboard_ee"}:
            from .processor import MapDeltaActionToURPose

            return [MapDeltaActionToURPose()]
        return []

    def is_policy_feature(self, key: str, value: object) -> bool:
        return isinstance(value, tuple) or value is float

    @property
    def is_connected(self) -> bool:
        rtde_connected = self.rtde_control is not None and self.rtde_receive is not None and self.rtde_io is not None
        return rtde_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.rtde_control = RTDEControlInterface(self.config.ip)
        self.rtde_receive = RTDEReceiveInterface(self.config.ip)
        self.rtde_io = RTDEIOInterface(self.config.ip)

        for cam in self.cameras.values():
            cam.connect()

        if calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("UR robots do not require LeRobot motor calibration.")

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        actual_q = self.rtde_receive.getActualQ()
        tcp_pose = self.rtde_receive.getActualTCPPose()

        obs_dict: RobotObservation = {
            **{f"{joint}.pos": float(value) for joint, value in zip(JOINT_NAMES, actual_q, strict=True)},
            **{key: float(value) for key, value in zip(EE_POSE_KEYS, tcp_pose, strict=True)},
            GRIPPER_OPEN: self._read_gripper_open(),
        }

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.read_latest()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        current_pose = {
            key: float(value)
            for key, value in zip(EE_POSE_KEYS, self.rtde_receive.getActualTCPPose(), strict=True)
        }
        goal_pose = {key: float(action.get(key, current_pose[key])) for key in EE_POSE_KEYS}

        if self.config.max_relative_target is not None:
            goal_present_pose = {key: (goal_pose[key], current_pose[key]) for key in EE_POSE_KEYS}
            goal_pose = ensure_safe_goal_position(goal_present_pose, self.config.max_relative_target)

        goal_pose = self._clip_to_workspace(goal_pose)

        pose_list = [goal_pose[key] for key in EE_POSE_KEYS]
        if self.config.command_mode == "servoL":
            self.rtde_control.servoL(
                pose_list,
                self.config.speed,
                self.config.acceleration,
                self.config.servo_time_s,
                self.config.lookahead_time_s,
                self.config.gain,
            )
        else:
            self.rtde_control.moveL(pose_list, self.config.speed, self.config.acceleration)

        sent_action: RobotAction = dict(goal_pose)
        if GRIPPER_OPEN in action:
            sent_action[GRIPPER_OPEN] = self._send_gripper_action(float(action[GRIPPER_OPEN]))

        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self.rtde_control.servoStop()
        except Exception as e:
            logger.debug(f"Failed to stop UR servo mode during disconnect: {e}")

        try:
            self.rtde_control.stopScript()
        except Exception as e:
            logger.debug(f"Failed to stop UR RTDE script during disconnect: {e}")

        for cam in self.cameras.values():
            cam.disconnect()

        self.rtde_control = None
        self.rtde_receive = None
        self.rtde_io = None
        logger.info(f"{self} disconnected.")

    def _send_gripper_action(self, gripper_open: float) -> float:
        open_value = 1.0 if gripper_open >= 0.5 else 0.0
        digital_state = self.config.gripper_open_state if open_value else not self.config.gripper_open_state
        self.rtde_io.setStandardDigitalOut(self.config.gripper_digital_output, digital_state)
        return open_value

    def _clip_to_workspace(self, pose: dict[str, float]) -> dict[str, float]:
        clipped_pose = dict(pose)
        for key, bounds in self.config.workspace_bounds.items():
            if key not in clipped_pose:
                raise ValueError(f"workspace_bounds contains unknown pose key '{key}'")
            lower, upper = bounds
            if lower > upper:
                raise ValueError(f"workspace_bounds for '{key}' has lower bound greater than upper bound")
            clipped_pose[key] = min(max(clipped_pose[key], lower), upper)
        return clipped_pose

    def _read_gripper_open(self) -> float:
        try:
            digital_state = self.rtde_receive.getDigitalOutState(self.config.gripper_digital_output)
        except AttributeError:
            return 0.0
        return 1.0 if bool(digital_state) == self.config.gripper_open_state else 0.0
