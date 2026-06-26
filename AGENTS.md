This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Local Project Focus: UR + LeRobot

For this checkout, prioritize Universal Robots (UR) workflows. The main robot type is `ur_follower`, implemented with `ur_rtde`.

- **Primary robot**: `--robot.type=ur_follower`
- **Core files**: `src/lerobot/robots/ur_follower/`, `docs/source/ur.mdx`, `tests/robots/test_ur_follower.py`
- **Dependency extra**: `lerobot[ur]` (`ur-rtde`), declared in `pyproject.toml`
- **Control interface**: policy/teleop actions are TCP pose keys `ee.x`, `ee.y`, `ee.z`, `ee.rx`, `ee.ry`, `ee.rz`, plus `gripper.open`
- **Observations**: keep TCP pose, six joint positions, `gripper.open`, and configured cameras. Joint positions are retained for state/diagnostics even though UR is controlled in TCP pose space.
- **Gripper**: `gripper.open` maps to standard digital output `0` by default. Use `--robot.gripper_digital_output=<N>` and `--robot.gripper_open_state=<bool>` if the wiring differs.
- **Teleoperation**: `gamepad` and `keyboard_ee` produce `delta_x/y/z/gripper`; the default processor maps those deltas to UR TCP pose actions via `MapDeltaActionToURPose`.
- **Safety defaults**: preserve conservative `max_relative_target`, `speed`, `acceleration`, and `workspace_bounds` behavior. Do not add connection-time motion or automatic homing.
- **Calibration**: UR does not use LeRobot motor calibration. `lerobot-calibrate` should remain a no-op-style connect/report/disconnect path for UR.

When modifying UR behavior, keep the same abstraction boundary as other robots: `ur_rtde` must stay encapsulated inside `URFollower`; scripts and processors should interact through `Robot`, `RobotConfig`, and processor pipelines.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

UR-specific validation:

```bash
uv run --extra test pytest tests/robots/test_ur_follower.py -q
uv run --extra dev ruff check src/lerobot/robots/ur_follower src/lerobot/processor/factory.py src/lerobot/rollout/context.py tests/robots/test_ur_follower.py
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
