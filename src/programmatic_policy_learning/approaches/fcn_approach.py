"""Fully convolutional behavioral cloning baseline for grid tasks."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import MultiDiscrete
from gym.spaces import MultiDiscrete as LegacyMultiDiscrete
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from programmatic_policy_learning.approaches.base_approach import BaseApproach
from programmatic_policy_learning.approaches.lpp_utils.utils import (
    infer_episode_success,
    run_single_episode,
)
from programmatic_policy_learning.data.collect import get_demonstrations
from programmatic_policy_learning.data.demo_types import Trajectory


class _FCNPolicy(nn.Module):
    """Small FCN matching the paper's receptive-field-oriented baseline."""

    def __init__(
        self,
        in_channels: int,
        *,
        num_conv_layers: int = 8,
        input_hidden_channels: int = 8,
        hidden_channels: int = 4,
    ) -> None:
        super().__init__()
        if num_conv_layers <= 0:
            raise ValueError("num_conv_layers must be positive.")

        layers: list[nn.Module] = []
        current_channels = in_channels
        for layer_idx in range(num_conv_layers):
            next_channels = (
                input_hidden_channels if layer_idx == 0 else hidden_channels
            )
            layers.append(
                nn.Conv2d(
                    current_channels,
                    next_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            current_channels = next_channels
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(current_channels, 1, kernel_size=1, stride=1)

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Return per-cell logits for one or more observations."""
        features = self.backbone(obs_tensor)
        return self.head(features).squeeze(1)


class FCNApproach(BaseApproach[np.ndarray, tuple[int, int]]):
    """Behavioral cloning baseline using a deep fully convolutional network."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Any,
        action_space: Any,
        seed: int,
        expert: Any,
        env_factory: Callable[[int], gym.Env],
        *,
        demo_numbers: Sequence[int] = tuple(range(10)),
        object_types: Sequence[str] | None = None,
        max_demo_length: int = 100,
        num_epochs: int = 200,
        min_epochs: int = 25,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        num_conv_layers: int = 8,
        input_hidden_channels: int = 8,
        hidden_channels: int = 4,
        device: str = "auto",
        train_before_eval: bool = True,
    ) -> None:
        super().__init__(environment_description, observation_space, action_space, seed)
        if not isinstance(action_space, (MultiDiscrete, LegacyMultiDiscrete)):
            raise TypeError(
                "FCNApproach requires a MultiDiscrete action space. "
                f"Received {type(action_space)!r}."
            )
        self._expert = expert
        self._env_factory = env_factory
        self._demo_numbers = tuple(int(each) for each in demo_numbers)
        self._object_types = tuple(str(each) for each in (object_types or ()))
        self._max_demo_length = int(max_demo_length)
        self._num_epochs = int(num_epochs)
        self._min_epochs = int(min_epochs)
        self._batch_size = int(batch_size)
        self._learning_rate = float(learning_rate)
        self._weight_decay = float(weight_decay)
        self._num_conv_layers = int(num_conv_layers)
        self._input_hidden_channels = int(input_hidden_channels)
        self._hidden_channels = int(hidden_channels)
        self._device = self._resolve_device(device)
        self._train_before_eval = bool(train_before_eval)
        self._training_seed = int(seed)
        self._last_observation: np.ndarray | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._token_to_idx: dict[str, int] = {}
        self._idx_to_token: list[str] = []
        self._model: _FCNPolicy | None = None
        self._is_trained = False
        self.training_summary: dict[str, Any] = {}
        self._seed_everything()

    def _seed_everything(self) -> None:
        """Seed all RNGs used by the FCN training pipeline."""
        random.seed(self._training_seed)
        np.random.seed(self._training_seed)
        torch.manual_seed(self._training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._training_seed)
            torch.cuda.manual_seed_all(self._training_seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def _resolve_device(requested_device: str) -> torch.device:
        normalized = str(requested_device).strip().lower()
        if normalized == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(normalized)

    @staticmethod
    def _coerce_grid(obs: Any) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=object)
        if obs_arr.ndim != 2:
            raise ValueError(
                "FCNApproach expects 2D grid observations; "
                f"received shape {obs_arr.shape}."
            )
        return obs_arr

    @staticmethod
    def _flatten_action(action: tuple[int, int], grid_shape: tuple[int, int]) -> int:
        row, col = int(action[0]), int(action[1])
        height, width = grid_shape
        if row < 0 or row >= height or col < 0 or col >= width:
            raise ValueError(
                f"Action {action} is outside the grid bounds {grid_shape}."
            )
        return row * width + col

    @staticmethod
    def _unflatten_action(action_idx: int, grid_shape: tuple[int, int]) -> tuple[int, int]:
        height, width = grid_shape
        if action_idx < 0 or action_idx >= height * width:
            raise ValueError(
                f"Action index {action_idx} is outside the grid bounds {grid_shape}."
            )
        row, col = divmod(int(action_idx), width)
        return (row, col)

    def _ensure_action_shape_is_2d(self) -> None:
        nvec = np.asarray(self._action_space.nvec, dtype=int).reshape(-1)
        if nvec.size != 2:
            raise ValueError(
                "FCNApproach expects a 2D MultiDiscrete action space; "
                f"received {self._action_space}."
            )

    @staticmethod
    def _max_grid_shape(
        demonstrations: Sequence[Trajectory[np.ndarray, tuple[int, int]]],
    ) -> tuple[int, int]:
        max_height = 0
        max_width = 0
        for trajectory in demonstrations:
            for obs, _ in trajectory.steps:
                obs_arr = np.asarray(obs, dtype=object)
                if obs_arr.ndim != 2:
                    raise ValueError(
                        "FCNApproach expects 2D grid observations; "
                        f"received shape {obs_arr.shape}."
                    )
                max_height = max(max_height, int(obs_arr.shape[0]))
                max_width = max(max_width, int(obs_arr.shape[1]))
        if max_height <= 0 or max_width <= 0:
            raise ValueError("Could not infer a valid grid shape from demonstrations.")
        return (max_height, max_width)

    @staticmethod
    def _valid_mask(
        obs_shape: tuple[int, int], canvas_shape: tuple[int, int]
    ) -> np.ndarray:
        mask = np.zeros(canvas_shape, dtype=np.bool_)
        mask[: obs_shape[0], : obs_shape[1]] = True
        return mask

    def _build_vocabulary(
        self, demonstrations: Sequence[Trajectory[np.ndarray, tuple[int, int]]]
    ) -> None:
        ordered_tokens: list[str] = []
        seen = set()

        def add_token(raw_token: Any) -> None:
            token = str(raw_token)
            if token in seen:
                return
            seen.add(token)
            ordered_tokens.append(token)

        for token in self._object_types:
            add_token(token)
        for trajectory in demonstrations:
            for obs, _ in trajectory.steps:
                obs_arr = self._coerce_grid(obs)
                for token in obs_arr.flat:
                    add_token(token)
        add_token("__unk__")
        self._idx_to_token = ordered_tokens
        self._token_to_idx = {
            token: idx for idx, token in enumerate(self._idx_to_token)
        }

    def _encode_observation(
        self,
        obs: Any,
        *,
        canvas_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        if not self._token_to_idx:
            raise RuntimeError("Vocabulary is not initialized.")
        obs_arr = self._coerce_grid(obs)
        target_shape = canvas_shape
        if target_shape is None:
            target_shape = cast(tuple[int, int], tuple(int(x) for x in obs_arr.shape))
        if obs_arr.shape[0] > target_shape[0] or obs_arr.shape[1] > target_shape[1]:
            raise ValueError(
                "Observation shape exceeds the requested FCN canvas: "
                f"canvas={target_shape}, received={obs_arr.shape}."
            )
        encoded = np.zeros(
            (len(self._idx_to_token), *target_shape),
            dtype=np.float32,
        )
        unk_idx = self._token_to_idx["__unk__"]
        for row in range(obs_arr.shape[0]):
            for col in range(obs_arr.shape[1]):
                token_idx = self._token_to_idx.get(str(obs_arr[row, col]), unk_idx)
                encoded[token_idx, row, col] = 1.0
        return encoded

    def _collect_demonstrations(
        self,
    ) -> list[Trajectory[np.ndarray, tuple[int, int]]]:
        _, demo_dict = get_demonstrations(
            self._env_factory,
            self._expert,
            self._demo_numbers,
            max_demo_length=self._max_demo_length,
        )
        return [
            cast(Trajectory[np.ndarray, tuple[int, int]], demo_dict[demo_id])
            for demo_id in self._demo_numbers
        ]

    def train_offline(self) -> None:
        """Collect expert demonstrations and fit the FCN by behavior cloning."""
        demonstrations = self._collect_demonstrations()
        self.set_demonstrations(cast(list[Trajectory[Any, Any]], demonstrations))
        if not demonstrations:
            raise ValueError("No demonstrations were collected for FCN training.")

        self._ensure_action_shape_is_2d()
        self._grid_shape = self._max_grid_shape(demonstrations)
        self._build_vocabulary(demonstrations)

        encoded_obs: list[np.ndarray] = []
        action_labels: list[int] = []
        valid_masks: list[np.ndarray] = []
        observed_shapes: list[tuple[int, int]] = []
        for trajectory in demonstrations:
            for obs, action in trajectory.steps:
                obs_arr = self._coerce_grid(obs)
                encoded_obs.append(
                    self._encode_observation(obs, canvas_shape=self._grid_shape)
                )
                observed_shapes.append(
                    cast(tuple[int, int], tuple(int(x) for x in obs_arr.shape))
                )
                valid_masks.append(self._valid_mask(observed_shapes[-1], self._grid_shape))
                action_labels.append(self._flatten_action(action, self._grid_shape))

        obs_tensor = torch.from_numpy(np.stack(encoded_obs, axis=0))
        mask_tensor = torch.from_numpy(np.stack(valid_masks, axis=0))
        label_tensor = torch.tensor(action_labels, dtype=torch.long)
        dataset = TensorDataset(obs_tensor, mask_tensor, label_tensor)
        batch_size = min(max(1, self._batch_size), len(dataset))
        data_loader_generator = torch.Generator()
        data_loader_generator.manual_seed(self._training_seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=data_loader_generator,
        )

        self._model = _FCNPolicy(
            in_channels=obs_tensor.shape[1],
            num_conv_layers=self._num_conv_layers,
            input_hidden_channels=self._input_hidden_channels,
            hidden_channels=self._hidden_channels,
        ).to(self._device)

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        obs_tensor = obs_tensor.to(self._device)
        mask_tensor = mask_tensor.to(self._device)
        label_tensor = label_tensor.to(self._device)
        best_accuracy = 0.0
        epochs_trained = 0
        final_loss = 0.0

        for epoch_idx in range(self._num_epochs):
            assert self._model is not None
            self._model.train()
            total_loss = 0.0
            total_examples = 0
            for batch_obs, batch_masks, batch_labels in loader:
                batch_obs = batch_obs.to(self._device)
                batch_masks = batch_masks.to(self._device)
                batch_labels = batch_labels.to(self._device)
                optimizer.zero_grad()
                logits = self._model(batch_obs).reshape(batch_obs.shape[0], -1)
                logits = logits.masked_fill(
                    ~batch_masks.reshape(batch_masks.shape[0], -1),
                    -1e9,
                )
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                batch_size_actual = int(batch_obs.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size_actual
                total_examples += batch_size_actual

            self._model.eval()
            with torch.no_grad():
                logits = self._model(obs_tensor).reshape(obs_tensor.shape[0], -1)
                logits = logits.masked_fill(
                    ~mask_tensor.reshape(mask_tensor.shape[0], -1),
                    -1e9,
                )
                predictions = torch.argmax(logits, dim=1)
                accuracy = float((predictions == label_tensor).float().mean().cpu())
            epochs_trained = epoch_idx + 1
            final_loss = total_loss / max(1, total_examples)
            best_accuracy = max(best_accuracy, accuracy)
            logging.info(
                "FCN epoch %d/%d: loss=%.6f accuracy=%.4f",
                epoch_idx + 1,
                self._num_epochs,
                final_loss,
                accuracy,
            )
            if accuracy >= 0.999 and epochs_trained >= self._min_epochs:
                break

        self.training_summary = {
            "num_demos": len(demonstrations),
            "num_training_steps": len(dataset),
            "grid_shape": list(self._grid_shape),
            "observed_grid_shapes": [list(shape) for shape in sorted(set(observed_shapes))],
            "vocabulary_size": len(self._idx_to_token),
            "epochs_requested": self._num_epochs,
            "epochs_trained": epochs_trained,
            "final_loss": final_loss,
            "best_accuracy": best_accuracy,
            "device": str(self._device),
            "demo_numbers": list(self._demo_numbers),
        }
        self._is_trained = True

    def _predict_action(self, obs: Any) -> tuple[int, int]:
        if self._model is None or not self._is_trained or self._grid_shape is None:
            raise RuntimeError("FCNApproach must be trained before prediction.")
        obs_arr = self._coerce_grid(obs)
        obs_shape = cast(tuple[int, int], tuple(int(x) for x in obs_arr.shape))
        encoded = self._encode_observation(obs, canvas_shape=obs_shape)
        obs_tensor = (
            torch.from_numpy(encoded)
            .unsqueeze(0)
            .to(self._device)
        )
        self._model.eval()
        with torch.no_grad():
            logits = self._model(obs_tensor).reshape(1, -1)
            action_idx = int(torch.argmax(logits, dim=1).item())
        return self._unflatten_action(action_idx, obs_shape)

    def reset(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        del info
        if self._train_before_eval and not self._is_trained:
            self.train_offline()
        self._last_observation = self._coerce_grid(obs)

    def update(
        self, obs: np.ndarray, reward: float, done: bool, info: dict[str, Any]
    ) -> None:
        del reward, done, info
        self._last_observation = self._coerce_grid(obs)

    def _get_action(self) -> tuple[int, int]:
        if self._last_observation is None:
            raise RuntimeError("Call reset() before requesting an action.")
        if not self._is_trained:
            self.train_offline()
        return self._predict_action(self._last_observation)

    def test_policy_on_envs(
        self,
        test_env_nums: Iterable[int] = range(11, 20),
        max_num_steps: int = 50,
        *,
        base_class_name: str | None = None,
        _record_videos: bool = False,
        _video_format: str = "mp4",
        **_extra_env_kwargs: Any,
    ) -> list[bool]:
        """Evaluate the FCN policy on held-out env instances."""
        if not self._is_trained:
            self.train_offline()
        results: list[bool] = []
        for env_num in test_env_nums:
            env = self._env_factory(int(env_num))
            reward, terminated, final_info = run_single_episode(
                env,
                self,
                max_num_steps=int(max_num_steps),
                reset_seed=int(env_num),
            )
            results.append(
                infer_episode_success(
                    reward=float(reward),
                    terminated=bool(terminated),
                    action_mode="discrete",
                    base_class_name=base_class_name,
                    final_info=final_info,
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        """Persist the learned weights and token vocabulary."""
        if self._model is None or self._grid_shape is None:
            raise RuntimeError("No trained FCN model is available to save.")
        payload = {
            "state_dict": self._model.state_dict(),
            "grid_shape": self._grid_shape,
            "token_to_idx": self._token_to_idx,
            "idx_to_token": self._idx_to_token,
            "training_summary": self.training_summary,
        }
        torch.save(payload, str(path))
