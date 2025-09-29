from typing import List, Sequence, Tuple, Union

import torch
from torch import nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.plain_conv_encoder import StackedConvBlocks


class SpatialSelfAttention(nn.Module):
    """Applies multi-head self-attention across spatial positions of a feature map."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        num_heads = max(1, num_heads)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            return x
        b, c = x.shape[:2]
        spatial_dims: Sequence[int] = x.shape[2:]
        flattened = x.view(b, c, -1).permute(0, 2, 1)
        attended, _ = self.attention(flattened, flattened, flattened, need_weights=False)
        attended = self.norm(attended)
        attended = attended.permute(0, 2, 1).contiguous().view(b, c, *spatial_dims)
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
        self.attention_block = SpatialSelfAttention(bottleneck_channels, attention_heads)
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
