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

from dataclasses import dataclass

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, RobotAction, RobotObservation, TransitionKey

from .ur_follower import EE_POSE_KEYS, GRIPPER_OPEN


@ProcessorStepRegistry.register("map_delta_action_to_ur_pose")
@dataclass
class MapDeltaActionToURPose(ProcessorStep):
    """Map gamepad/keyboard end-effector deltas to UR absolute TCP pose actions."""

    position_scale: float = 0.01
    noise_threshold: float = 1e-3

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if action is None or not isinstance(action, dict):
            raise ValueError(f"Action should be a RobotAction type (dict), but got {type(action)}")
        if observation is None or not isinstance(observation, dict):
            raise ValueError("MapDeltaActionToURPose requires a robot observation.")

        new_transition[TransitionKey.ACTION] = self.action(action.copy(), observation)
        return new_transition

    def action(self, action: RobotAction, observation: RobotObservation) -> RobotAction:
        pose = {key: float(observation[key]) for key in EE_POSE_KEYS}
        delta_x = float(action.get("delta_x", 0.0))
        delta_y = float(action.get("delta_y", 0.0))
        delta_z = float(action.get("delta_z", 0.0))

        if (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5 > self.noise_threshold:
            pose["ee.x"] += delta_x * self.position_scale
            pose["ee.y"] += delta_y * self.position_scale
            pose["ee.z"] += delta_z * self.position_scale

        pose[GRIPPER_OPEN] = self._map_gripper(action.get("gripper"), observation.get(GRIPPER_OPEN, 0.0))
        return pose

    def _map_gripper(self, gripper: float | None, current_open: float) -> float:
        if gripper is None:
            return float(current_open)
        if gripper > 1.5:
            return 1.0
        if gripper < 0.5:
            return 0.0
        return float(current_open)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for key in ["delta_x", "delta_y", "delta_z", "gripper"]:
            action_features.pop(key, None)
        for key in EE_POSE_KEYS:
            action_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        action_features[GRIPPER_OPEN] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features
