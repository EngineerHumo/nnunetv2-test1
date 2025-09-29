from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

try:
    from visdom import Visdom  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Visdom = None  # type: ignore


TensorLike = Union[torch.Tensor, np.ndarray]


class VisdomLogger:
    """Utility that mirrors selected training artefacts to a Visdom dashboard."""

    def __init__(
        self,
        dataset_name: str,
        env_suffix: str = "",
        enable: bool = True,
        port: Optional[int] = None,
    ) -> None:
        env_name = f"nnUNet_{dataset_name}"
        if env_suffix:
            env_name = f"{env_name}_{env_suffix}"

        self.enabled = bool(enable and Visdom is not None)
        self._train_logged_epoch: Optional[int] = None
        self._val_logged_epoch: Optional[int] = None
        self._vis: Optional[Visdom] = None
        self._wins: dict[str, str] = {}

        if self.enabled:
            try:
                self._vis = Visdom(env=env_name, port=port)
                if hasattr(self._vis, "check_connection") and not self._vis.check_connection():
                    raise RuntimeError("Visdom server is not reachable")
            except Exception as exc:  # pragma: no cover - only triggered when Visdom fails
                print(f"[Visdom] Disabled: {exc}")
                self.enabled = False
                self._vis = None

    def on_epoch_start(self, epoch: int) -> None:
        if not self.enabled:
            return
        self._train_logged_epoch = None
        self._val_logged_epoch = None

    def log_training_batch(self, data: TensorLike | Sequence[TensorLike], target: TensorLike | Sequence[TensorLike],
                           output: TensorLike | Sequence[TensorLike], epoch: int) -> None:
        if not self.enabled or self._train_logged_epoch == epoch:
            return
        image = self._prepare_input_image(data)
        label = self._prepare_segmentation(target, from_one_hot=False)
        prediction = self._prepare_segmentation(output, from_one_hot=True)
        self._publish_image("train/input", image, f"Epoch {epoch} – train input")
        self._publish_image("train/label", label, f"Epoch {epoch} – train label (grayscale)")
        self._publish_image("train/prediction", prediction, f"Epoch {epoch} – train prediction (grayscale)")
        self._train_logged_epoch = epoch

    def log_validation_batch(self, data: TensorLike | Sequence[TensorLike],
                             target: TensorLike | Sequence[TensorLike],
                             prediction_one_hot: TensorLike | Sequence[TensorLike],
                             epoch: int) -> None:
        if not self.enabled or self._val_logged_epoch == epoch:
            return
        image = self._prepare_input_image(data)
        label = self._prepare_segmentation(target, from_one_hot=False)
        prediction = self._prepare_segmentation(prediction_one_hot, from_one_hot=True)
        self._publish_image("val/input", image, f"Epoch {epoch} – val input")
        self._publish_image("val/label", label, f"Epoch {epoch} – val label (grayscale)")
        self._publish_image("val/prediction", prediction, f"Epoch {epoch} – val prediction (grayscale)")
        self._val_logged_epoch = epoch

    def _publish_image(self, name: str, image: Optional[np.ndarray], caption: str) -> None:
        if not self.enabled or image is None or self._vis is None:
            return
        try:
            self._vis.image(image, win=name, opts={"caption": caption})
        except Exception as exc:  # pragma: no cover - depends on visdom runtime
            print(f"[Visdom] Failed to publish '{name}': {exc}")
            self.enabled = False

    @staticmethod
    def _select_tensor(tensor: TensorLike | Sequence[TensorLike]) -> Optional[torch.Tensor]:
        if isinstance(tensor, (list, tuple)):
            if len(tensor) == 0:
                return None
            tensor = tensor[0]
        if tensor is None:
            return None
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        assert isinstance(tensor, torch.Tensor)
        return tensor

    def _prepare_input_image(self, tensor: TensorLike | Sequence[TensorLike]) -> Optional[np.ndarray]:
        t = self._select_tensor(tensor)
        if t is None:
            return None
        t = t.detach().float().cpu()
        if t.ndim >= 4:
            t = t[0]
        if t.ndim == 3:
            if t.shape[0] > 1:
                t = t.mean(0)
            else:
                t = t[0]
        if t.ndim != 2:
            return None
        arr = t.numpy()
        arr = arr - arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val
        return arr[np.newaxis, ...].astype(np.float32)

    def _prepare_segmentation(self, tensor: TensorLike | Sequence[TensorLike], *, from_one_hot: bool) -> Optional[np.ndarray]:
        t = self._select_tensor(tensor)
        if t is None:
            return None
        t = t.detach().float().cpu()
        if t.ndim >= 4:
            t = t[0]
        if from_one_hot:
            if t.ndim >= 3:
                seg = torch.argmax(t, dim=0)
            else:
                seg = t
        else:
            if t.ndim == 3 and t.shape[0] > 1:
                if torch.all((t == 0) | (t == 1)):
                    seg = torch.argmax(t, dim=0)
                else:
                    seg = t[0]
            elif t.ndim == 3 and t.shape[0] == 1:
                seg = t[0]
            elif t.ndim == 2:
                seg = t
            else:
                seg = t.squeeze()
        if seg.ndim != 2:
            seg = seg.squeeze()
        if seg.ndim != 2:
            return None
        if seg.max() > 0:
            seg = seg / seg.max()
        arr = seg.numpy().astype(np.float32)
        return arr[np.newaxis, ...]
