import torch
from torch import nn, Tensor
from einops import einsum, rearrange

from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


class TeamworkLora(nn.Module):
    def __init__(
        self, teammates: int, in_features: int, out_features: int, rank: int, bias: bool
    ):
        super().__init__()

        self.down = nn.Parameter(torch.empty([teammates, in_features, rank]))
        nn.init.normal_(self.down, mean=0.0, std=0.02)
        self.up = nn.Parameter(torch.empty([teammates, rank, out_features]))
        nn.init.zeros_(self.up)
        if bias:
            self.bias = nn.Parameter(torch.empty([teammates, out_features]))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor):
        x = einsum(x, self.down, "t ... i, t i r -> ... r")
        x = einsum(x, self.up, "... r, t r o -> t ... o")
        if self.bias is not None:
            b = self.bias
            while len(b.shape) < len(x.shape):
                b = b.unsqueeze(1)
            x = x + b
        return x


class TeamworkLinear(nn.Linear):
    def __init__(self, parent: nn.Linear, teammates: int, rank: int):
        assert isinstance(parent, nn.Linear), (
            f"attempted to add teamwork to {type(parent)}"
        )
        self.__dict__.update(parent.__dict__)
        self.adapter = TeamworkLora(
            teammates,
            self.in_features,
            self.out_features,
            rank,
            self.bias is not None,
        ).to(device=self.weight.device, dtype=self.weight.dtype)

    def forward(self, input: Tensor):
        return super().forward(input) + self.adapter(input)


class TeamworkConv2d(nn.Conv2d):
    def __init__(self, parent: nn.Conv2d, teammates: int, rank: int):
        self.__dict__.update(parent.__dict__)
        self.adapter = TeamworkLora(
            teammates,
            self.in_channels,
            self.out_channels,
            rank,
            self.bias is not None,
        ).to(device=self.weight.device, dtype=self.weight.dtype)

    def forward(self, input: Tensor):
        return super().forward(input) + rearrange(
            self.adapter(rearrange(input, "b c ... -> b ... c")), "b ... c -> b c ..."
        )


def adapt_flux(model: FluxTransformer2DModel, teammates: int, rank: int):
    for block in [*model.single_transformer_blocks, *model.transformer_blocks]:
        from diffusers.models.transformers.transformer_flux import (
            FluxTransformerBlock,
            FluxSingleTransformerBlock,
        )

        if isinstance(block, FluxTransformerBlock):
            block.norm1.linear = TeamworkLinear(block.norm1.linear, teammates, rank)
            block.ff.net[0].proj = TeamworkLinear(block.ff.net[0].proj, teammates, rank)
            block.ff.net[2] = TeamworkLinear(block.ff.net[2], teammates, rank)
        if isinstance(block, FluxSingleTransformerBlock):
            block.norm.linear = TeamworkLinear(block.norm.linear, teammates, rank)
            block.proj_mlp = TeamworkLinear(block.proj_mlp, teammates, rank)
        block.attn.to_k = TeamworkLinear(block.attn.to_k, teammates, rank)
        block.attn.to_q = TeamworkLinear(block.attn.to_q, teammates, rank)
        block.attn.to_v = TeamworkLinear(block.attn.to_v, teammates, rank)
        if hasattr(block.attn, "to_out") and block.attn.to_out is not None:
            block.attn.to_out[0] = TeamworkLinear(block.attn.to_out[0], teammates, rank)


def adapt_sd3(model: SD3Transformer2DModel, teammates: int, rank: int):
    for block in model.transformer_blocks:
        block.norm1.linear = TeamworkLinear(block.norm1.linear, teammates, rank)
        block.ff.net[0].proj = TeamworkLinear(block.ff.net[0].proj, teammates, rank)
        block.ff.net[2] = TeamworkLinear(block.ff.net[2], teammates, rank)
        block.attn.to_k = TeamworkLinear(block.attn.to_k, teammates, rank)
        block.attn.to_q = TeamworkLinear(block.attn.to_q, teammates, rank)
        block.attn.to_v = TeamworkLinear(block.attn.to_v, teammates, rank)
        block.attn.to_out[0] = TeamworkLinear(block.attn.to_out[0], teammates, rank)


def adapt_sdxl(model: UNet2DConditionModel, teammates: int, rank: int):
    for block in [*model.down_blocks, model.mid_block, *model.up_blocks]:
        if hasattr(block, "attentions"):
            for attns in block.attentions:
                for attn in attns.transformer_blocks:
                    attn.ff.net[0].proj = TeamworkLinear(
                        attn.ff.net[0].proj, teammates, rank
                    )
                    attn.ff.net[2] = TeamworkLinear(attn.ff.net[2], teammates, rank)
                    attn.attn1.to_k = TeamworkLinear(attn.attn1.to_k, teammates, rank)
                    attn.attn1.to_q = TeamworkLinear(attn.attn1.to_q, teammates, rank)
                    attn.attn1.to_v = TeamworkLinear(attn.attn1.to_v, teammates, rank)
                    attn.attn1.to_out[0] = TeamworkLinear(
                        attn.attn1.to_out[0], teammates, rank
                    )
        for resnet in block.resnets:
            resnet.conv1 = TeamworkConv2d(resnet.conv1, teammates, rank)
            resnet.conv2 = TeamworkConv2d(resnet.conv2, teammates, rank)
            if resnet.conv_shortcut is not None:
                resnet.conv_shortcut = TeamworkConv2d(
                    resnet.conv_shortcut, teammates, rank
                )
