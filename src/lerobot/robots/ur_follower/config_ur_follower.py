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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("ur_follower")
@dataclass
class URFollowerConfig(RobotConfig):
    """Configuration for Universal Robots arms controlled with ur_rtde."""

    ip: str

    # RTDE command mode. `servoL` is intended for continuous control loops; `moveL`
    # is useful for slower replay/debug flows.
    command_mode: str = "servoL"

    # TCP pose action safety limit. A scalar applies to all pose dimensions, or a
    # dict can specify per-key limits for ee.x/y/z/rx/ry/rz.
    max_relative_target: float | dict[str, float] | None = 0.02

    # ur_rtde linear move/servo parameters.
    speed: float = 0.25
    acceleration: float = 0.5
    servo_time_s: float = 0.008
    lookahead_time_s: float = 0.1
    gain: int = 300

    # Digital output used for the gripper. Action key `gripper.open` is converted
    # to a boolean state on this output.
    gripper_digital_output: int = 0
    gripper_open_state: bool = True

    # Optional TCP workspace limits. Keys are ee.x/ee.y/ee.z/ee.rx/ee.ry/ee.rz.
    workspace_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.command_mode not in {"servoL", "moveL"}:
            raise ValueError("command_mode must be either 'servoL' or 'moveL'")
