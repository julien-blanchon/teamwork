from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np
from random import randint
from typing import Callable
import PIL.Image


@dataclass(frozen=True)
class Selection:
    teammate_indices: Tensor
    input_subindices: Tensor  # (input_components,): int
    output_subindices: Tensor  # (output_components,): int


@dataclass
class ComponentInBatch:
    teammate: int
    component: str
    image: Tensor | None
    extra: Tensor | None
    weight: Tensor | None
    cropinfo: Tensor | None
    output: bool


def image_pt2np(image_pt: Tensor):
    image_pt = image_pt * 0.5 + 0.5  # Scale to [0, 1] range
    image_np = image_pt.cpu().float().permute(1, 2, 0).numpy()
    image_np = image_np.clip(0, 1)
    return image_np


def image_np2pil(image_np: np.ndarray):
    if image_np.dtype != np.uint8:
        image_np = (image_np.clip(0, 1) * 255).astype(np.uint8)
    return PIL.Image.fromarray(image_np)


def image_pt2pil(image_pt: Tensor):
    return image_np2pil(image_pt2np(image_pt))


class BatchBuilder:
    """
    Teamwork requires arranging input images and empty outputs into a batch, with information
    about which batch components corrispond to which model teammates. That is really a pain
    to do in each pipeline, so here is a shared helper class. It also handles all the usual
    image processing logic.
    """

    def __init__(self, teammates: list[str], device: torch.device, dtype: torch.dtype):
        self.teammates = teammates
        self.device = device
        self.dtype = dtype
        self.components: list[ComponentInBatch] = []
        self.resolution: tuple[int, int] | None = None

    def provide(
        self,
        component: str,
        image: np.ndarray | Tensor | PIL.Image.Image,
        weight: np.ndarray | Tensor | None = None,
        extra: np.ndarray | Tensor | None = None,
        cropinfo: np.ndarray | None = None,
        kind: str | None = None,
        image_format: str = "auto",
    ):
        """
        Provide a known image, either as input or to compute output loss.
        """

        if isinstance(image, PIL.Image.Image):
            image_np = np.array(image.convert("RGB"))
            image_tensor = torch.from_numpy(image_np)
            if image_format == "auto":
                image_format = "np"
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.expand_dims(image, 2)
            image_tensor = torch.from_numpy(image)
            if image_format == "auto":
                image_format = "np"
        elif isinstance(image, torch.Tensor):
            image_tensor = image
            if image_format == "auto":
                image_format = "pt"
        else:
            raise ValueError(
                f"image type {type(image)} is not a pil image, numpy array, or torch tensor"
            )

        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.to(torch.float32) / 255.0
        elif image_tensor.dtype == torch.uint16:
            image_tensor = image_tensor.to(torch.float32) / 65535.0
        else:
            pass

        if image_format == "np":
            image_tensor = image_tensor * 2 - 1
            image_tensor = image_tensor.permute(2, 0, 1)

        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)

        if image_tensor.shape[0] == 4:
            image_tensor = image_tensor[:3, :, :]

        image_tensor = image_tensor.nan_to_num()

        # Handle weight if present
        weight_tensor = None
        if weight is not None:
            if isinstance(weight, torch.Tensor):
                weight_tensor = weight.to(self.device, self.dtype)
            else:
                weight_tensor = torch.from_numpy(weight).to(self.device, self.dtype)
            if len(weight_tensor.shape) == 3:
                assert weight_tensor.shape[-1] == 1 or weight_tensor.shape[-1] == 3
                weight_tensor = weight_tensor[:, :, 1]

        # Handle concat channels if present
        concat_tensor = None
        if extra is not None:
            if isinstance(extra, torch.Tensor):
                concat_tensor = extra.to(self.device, self.dtype)
                assert len(concat_tensor.shape) == 3, (
                    f"concat tensor should have C,H,W (had shape {concat_tensor.shape})"
                )
            else:
                concat_tensor = torch.from_numpy(extra).to(self.device, self.dtype)
                assert len(concat_tensor.shape) == 3, (
                    f"concat ndarray should have H,W,C (had shape {concat_tensor.shape})"
                )
                concat_tensor = concat_tensor.permute(2, 0, 1)

        # Handle cropinfo if present
        cropinfo_tensor = None
        if cropinfo is not None:
            cropinfo_tensor = torch.from_numpy(cropinfo).to(self.device, self.dtype)

        _, h, w = image_tensor.shape
        if self.resolution is None:
            self.resolution = (h, w)
        else:
            assert self.resolution == (h, w), (
                f"component size {(h, w)} is inconsistant, was previously {self.resolution}"
            )

        if weight_tensor is not None:
            assert weight_tensor.shape == self.resolution, (
                f"weight shape {weight_tensor.shape} should be {self.resolution}"
            )

        if concat_tensor is not None:
            assert concat_tensor.shape[1:] == self.resolution, (
                f"concat shape {concat_tensor.shape} should have resolution {self.resolution}"
            )

        # Find matching teammate
        for idx, teammate in enumerate(self.teammates):
            teammate_component, teammate_kind = teammate.split(".")
            if component == teammate_component and (
                kind == teammate_kind or kind is None
            ):
                self.components.append(
                    ComponentInBatch(
                        teammate=idx,
                        component=component,
                        image=image_tensor,
                        extra=concat_tensor,
                        weight=weight_tensor,
                        cropinfo=cropinfo_tensor,
                        output=(teammate_kind != "in"),
                    )
                )

    def request_all(self, width: int | None = None, height: int | None = None):
        for teammate in self.teammates:
            teammate_component, teammate_kind = teammate.split(".")
            if teammate_kind == "out":
                self.request(teammate_component, width=width, height=height)

    def request(
        self, *components: str, width: int | None = None, height: int | None = None
    ):
        """
        Request that the model produce some outputs and include a placeholder. The resolution
        is infered from any previously provided input image, and otherwise.
        """

        if self.resolution is None:
            assert width is not None, "resolution must be known or provided"
            self.resolution = (height or width, width)
        if height is not None:
            assert self.resolution[0] == height
        if width is not None:
            assert self.resolution[1] == width

        for component in components:
            # Find matching teammate for output
            for idx, teammate in enumerate(self.teammates):
                teammate_component, teammate_kind = teammate.split(".")
                if teammate_component == component and teammate_kind == "out":
                    if any(existing.teammate == idx for existing in self.components):
                        continue
                    self.components.append(
                        ComponentInBatch(
                            teammate=idx,
                            component=component,
                            image=None,
                            extra=None,
                            weight=None,
                            cropinfo=None,
                            output=True,
                        )
                    )

    def split_io(self) -> tuple["BatchBuilder", "BatchBuilder"]:
        inputs = BatchBuilder(self.teammates, self.device, self.dtype)
        inputs.resolution = self.resolution
        outputs = BatchBuilder(self.teammates, self.device, self.dtype)
        outputs.resolution = self.resolution
        for component in self.components:
            if component.output:
                outputs.components.append(component)
            else:
                inputs.components.append(component)
        return (inputs, outputs)

    def crop_to_square(self):
        assert self.resolution is not None, "resolution must be known"
        h, w = self.resolution
        s = min(h, w)
        i = randint(0, h - s)
        j = randint(0, w - s)
        self.resolution = (s, s)
        for component in self.components:
            if component.image is not None:
                component.image = component.image[:, i : i + s, j : j + s]
            if component.weight is not None:
                component.weight = component.weight[i : i + s, j : j + s]
            if component.extra is not None:
                component.extra = component.extra[:, i : i + s, j : j + s]

    def pack_images(self, images: dict[str, Tensor], channels: int, scale: float = 1):
        for c in self.components:
            if c.component in images:
                found_channels = images[c.component].shape[0]
                assert found_channels == channels, (
                    f"found image with {found_channels} channels, expected {channels}"
                )

        to_pack = []
        for c in self.components:
            if c.component in images:
                to_pack.append(images[c.component].to(self.device, self.dtype))
            else:
                assert self.resolution is not None
                h, w = self.resolution
                zeros = torch.zeros(
                    (channels, int(h * scale), int(w * scale)),
                    device=self.device,
                    dtype=self.dtype,
                )
                to_pack.append(zeros)

        return (
            torch.stack(to_pack)
            if to_pack
            else torch.empty(0, device=self.device, dtype=self.dtype)
        )

    def packed_images(self, channels: int) -> Tensor:
        return self.pack_images(
            {c.component: c.image for c in self.components if c.image is not None},
            channels=channels,
        )

    def packed_encoded_images(
        self,
        encode: Callable[[Tensor], Tensor],
        channels: int,
        scale: float,
    ) -> Tensor:
        return self.pack_images(
            {
                c.component: encode(c.image.unsqueeze(0)).squeeze(0)
                for c in self.components
                if c.image is not None
            },
            channels=channels,
            scale=scale,
        )

    def packed_extra(self) -> Tensor | None:
        extra = []
        for component in self.components:
            if component.extra is not None:
                extra.append(component.extra.to(self.device, self.dtype))
            else:
                assert not extra, "only some components had extra channels"
                return None

        return (
            torch.stack(extra)
            if extra
            else torch.empty(0, device=self.device, dtype=self.dtype)
        )

    def packed_scaled_extra(self, scale: float) -> Tensor | None:
        extra = self.packed_extra()
        if extra is None or extra.numel() == 0:
            return extra

        assert self.resolution is not None
        h, w = self.resolution
        target_size = (int(h * scale), int(w * scale))

        # Interpolate (already has channel dimension)
        extra = torch.nn.functional.interpolate(
            extra,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return extra

    def packed_weights(self) -> Tensor:
        weights = []
        for component in self.components:
            if component.weight is not None:
                weights.append(component.weight.to(self.device, self.dtype))
            else:
                # For missing weights, the weight defaults to one
                assert self.resolution is not None
                zeros = torch.ones(
                    self.resolution, device=self.device, dtype=self.dtype
                )
                weights.append(zeros)

        return (
            torch.stack(weights)
            if weights
            else torch.empty(0, device=self.device, dtype=self.dtype)
        )

    def packed_scaled_weights(self, scale: float) -> Tensor:
        weights = self.packed_weights()
        if weights.numel() == 0:
            return weights

        assert self.resolution is not None
        h, w = self.resolution
        target_size = (int(h * scale), int(w * scale))

        # Add channel dimension for interpolation, then remove it
        weights = weights.unsqueeze(1)  # (batch, 1, h, w)
        weights = torch.nn.functional.interpolate(
            weights,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        weights = weights.squeeze(1)  # (batch, h', w')

        return weights

    def packed_cropinfo(self, default: Tensor) -> Tensor:
        cropinfos = []
        for component in self.components:
            if component.cropinfo is not None:
                cropinfos.append(component.cropinfo.to(self.device, default.dtype))
            else:
                cropinfos.append(default.to(self.device))

        if len(cropinfos) == 0:
            return torch.empty((0, 6), device=self.device, dtype=default.dtype)

        return torch.stack(cropinfos)

    def selection(self) -> Selection:
        teammate_indices = []
        input_indices = []
        output_indices = []

        for i, component in enumerate(self.components):
            teammate_indices.append(component.teammate)
            if component.output:
                output_indices.append(i)
            else:
                input_indices.append(i)

        return Selection(
            teammate_indices=torch.tensor(
                teammate_indices, dtype=torch.int64, device=self.device
            ),
            input_subindices=torch.tensor(
                input_indices, dtype=torch.int64, device=self.device
            ),
            output_subindices=torch.tensor(
                output_indices, dtype=torch.int64, device=self.device
            ),
        )

    def unpack_images(
        self, packed: Tensor, outputs_only: bool = False
    ) -> dict[str, Tensor]:
        unpacked = dict()
        for idx, component in enumerate(self.components):
            if component.output or not outputs_only:
                unpacked[component.component] = packed[idx]
        return unpacked

    def unpacked_images(self, outputs_only: bool = False) -> dict[str, Tensor]:
        return {
            c.component: c.image
            for c in self.components
            if c.image is not None and (c.output or not outputs_only)
        }

    @property
    def count(self):
        return len(self.components)
