from math import prod, sqrt
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class ExportFriendlyMultiheadAttention(nn.MultiheadAttention):
    """Multi-head attention variant that avoids unsupported ONNX ops.

    PyTorch's ``nn.MultiheadAttention`` dispatches to the fused
    ``aten._native_multi_head_attention`` operator which currently lacks an
    ONNX exporter lowering in opset <= 24.  When exporting with the traditional
    ``torch.onnx.export`` API this results in ``OpRegistrationError`` failures.

    To make the trained checkpoints portable we provide a light-weight
    re-implementation of the attention computation that relies purely on
    matrix multiplications and softmax – operators that are universally
    supported by ONNX.  The module subclasses ``nn.MultiheadAttention`` so that
    the state dict structure (``in_proj_weight``/``bias`` and the output
    projection parameters) matches the training checkpoints.

    The implementation covers the subset of features required by the
    ``SpatialSelfAttention`` block (batch-first inputs, self attention without
    masks).  If more advanced functionality is requested we fall back to the
    parent implementation to preserve correctness.
    """

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if (
            key_padding_mask is not None
            or attn_mask is not None
            or not self._qkv_same_embed_dim
            or not self.batch_first
            or is_causal
        ):
            # Defer to the upstream implementation when we encounter a setup
            # that requires features we do not explicitly handle here.
            return super().forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        b, seq_len, embed_dim = query.shape
        head_dim = embed_dim // self.num_heads
        if embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads for attention export")

        # Project query/key/value in a single linear operation to match the
        # layout used by ``nn.MultiheadAttention``.
        if self.in_proj_weight is None:
            raise RuntimeError("in_proj_weight is expected to be defined for self attention")

        q_proj = torch.nn.functional.linear(
            query,
            self.in_proj_weight[:embed_dim],
            None if self.in_proj_bias is None else self.in_proj_bias[:embed_dim],
        )
        k_proj = torch.nn.functional.linear(
            key,
            self.in_proj_weight[embed_dim : 2 * embed_dim],
            None if self.in_proj_bias is None else self.in_proj_bias[embed_dim : 2 * embed_dim],
        )
        v_proj = torch.nn.functional.linear(
            value,
            self.in_proj_weight[2 * embed_dim :],
            None if self.in_proj_bias is None else self.in_proj_bias[2 * embed_dim :],
        )

        scale = 1.0 / sqrt(head_dim)
        q = q_proj.view(b, seq_len, self.num_heads, head_dim).transpose(1, 2) * scale
        k = k_proj.view(b, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v_proj.view(b, seq_len, self.num_heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, seq_len, embed_dim)
        output = torch.nn.functional.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            weights = attn_probs
            if average_attn_weights:
                weights = weights.mean(dim=1)
            return output, weights

        return output, None

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.plain_conv_encoder import StackedConvBlocks


class SpatialSelfAttention(nn.Module):
    """Applies multi-head self-attention across spatial positions of a feature map."""

    def __init__(self, channels: int, num_heads: int = 4, max_tokens: int = 4096):
        super().__init__()
        num_heads = max(1, num_heads)
        self.attention = ExportFriendlyMultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(channels)
        self.max_tokens = max(1, int(max_tokens))

    @staticmethod
    def _reduced_shape(spatial_dims: Sequence[int], max_tokens: int) -> Tuple[int, ...]:
        target = list(spatial_dims)
        while target and prod(target) > max_tokens:
            largest_axis = max(range(len(target)), key=lambda i: target[i])
            target[largest_axis] = max(1, (target[largest_axis] + 1) // 2)
        return tuple(target)

    @staticmethod
    def _adaptive_pool(x: torch.Tensor, output_size: Tuple[int, ...]) -> torch.Tensor:
        if len(output_size) == 1:
            return F.adaptive_avg_pool1d(x, output_size[0])
        if len(output_size) == 2:
            return F.adaptive_avg_pool2d(x, output_size)
        if len(output_size) == 3:
            return F.adaptive_avg_pool3d(x, output_size)
        raise ValueError(f"Unsupported tensor rank for adaptive pooling: {x.ndim}")

    @staticmethod
    def _interpolate(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
        if len(size) == 1:
            return F.interpolate(x, size=size, mode="linear", align_corners=False)
        if len(size) == 2:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if len(size) == 3:
            return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        raise ValueError(f"Unsupported interpolation size: {size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            return x

        spatial_dims: Sequence[int] = x.shape[2:]
        target_dims = self._reduced_shape(spatial_dims, self.max_tokens)

        pooled = self._adaptive_pool(x, target_dims) if target_dims != tuple(spatial_dims) else x

        b, c = pooled.shape[:2]
        flattened = pooled.view(b, c, -1).permute(0, 2, 1)
        attended, _ = self.attention(flattened, flattened, flattened, need_weights=False)
        attended = self.norm(attended)
        attended = attended.permute(0, 2, 1).contiguous().view(b, c, *target_dims)

        if target_dims != tuple(spatial_dims):
            attended = self._interpolate(attended, spatial_dims)

        return attended


class AttentionUNet(PlainConvUNet):
    """PlainConvUNet variant with a spatial self-attention bottleneck."""

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op,
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, type] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, type] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, type] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        attention_heads: int = 4,
        attention_max_tokens: int = 4096,
    ):
        super().__init__(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            num_classes,
            n_conv_per_stage_decoder,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            deep_supervision,
            nonlin_first,
        )

        bottleneck_channels = self.encoder.output_channels[-1]
        self.attention_block = SpatialSelfAttention(
            bottleneck_channels,
            attention_heads,
            max_tokens=attention_max_tokens,
        )
        self.attention_fuse = StackedConvBlocks(
            num_convs=1,
            conv_op=self.encoder.conv_op,
            input_channels=bottleneck_channels * 2,
            output_channels=bottleneck_channels,
            kernel_size=self.encoder.kernel_sizes[-1],
            initial_stride=1,
            conv_bias=self.encoder.conv_bias,
            norm_op=self.encoder.norm_op,
            norm_op_kwargs=self.encoder.norm_op_kwargs,
            dropout_op=self.encoder.dropout_op,
            dropout_op_kwargs=self.encoder.dropout_op_kwargs,
            nonlin=self.encoder.nonlin,
            nonlin_kwargs=self.encoder.nonlin_kwargs,
            nonlin_first=nonlin_first,
        )

    def forward(self, x: torch.Tensor):
        skips = list(self.encoder(x))
        bottleneck = skips[-1]
        attended = self.attention_block(bottleneck)
        fused = torch.cat((bottleneck, attended), dim=1)
        skips[-1] = self.attention_fuse(fused)
        return self.decoder(skips)
