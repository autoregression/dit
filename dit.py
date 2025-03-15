import math
import typing

import einops
import torch


class DiTConfig(typing.NamedTuple):
    input_dimension: int = 3
    hidden_dimension: int = 256
    time_dimension: int = 1024
    heads: int = 4
    layers: int = 4
    patch_size: int = 1
    sequence_length: int = 1024
    eps: float = 1e-8


def zero_init(layer: torch.nn.Module) -> torch.nn.Module:
    torch.nn.init.zeros_(layer.weight)

    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)

    return layer


def create_position(hidden_dimension: int, heads: int, sequence_length: int) -> torch.Tensor:
    angle = torch.logspace(
        start=math.log10(0.5 * math.pi),
        end=math.log10(0.5 * math.pi * sequence_length),
        steps=(hidden_dimension // heads) // 4,
    )

    position = torch.arange(sequence_length, device=angle.device) / sequence_length
    position = torch.outer(angle, position)
    position = torch.polar(torch.ones_like(position), position)
    position = torch.stack([position.real, position.imag], dim=-1)

    return position


def apply_rope(x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    sequence_length = x.size(-2)
    x_rope, x_pass = x.chunk(2, dim=-1)
    x_rope = x_rope.float().reshape(*x_rope.shape[:-1], -1, 2)
    position = position[:sequence_length]
    position = position.view(1, 1, sequence_length, x_rope.size(-2), 2)

    x_rope = torch.stack(
        [
            x_rope[..., 0] * position[..., 0] - x_rope[..., 1] * position[..., 1],
            x_rope[..., 1] * position[..., 0] + x_rope[..., 0] * position[..., 1],
        ],
        dim=-1,
    )
    x_rope = x_rope.flatten(-2)

    return torch.cat([x_rope.type_as(x), x_pass], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (scale + 1) + shift


class TimeEmbedding(torch.nn.Module):
    def __init__(self, time_dimension: int) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(time_dimension, time_dimension, bias=False)
        self.register_buffer("scale", torch.randn([time_dimension // 2, 1]))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        x = 2 * math.pi * time.view(-1, 1) @ self.scale.T
        x = torch.cat([x.cos(), x.sin()], dim=-1)
        x = self.linear(x)

        return x[:, None, :]


class PatchEmbedding(torch.nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, patch_size: int) -> None:
        super().__init__()

        self.conv = torch.nn.Conv2d(input_dimension, hidden_dimension, patch_size, patch_size, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.conv(x)
        h, w = x.size(-2), x.size(-1)
        x = einops.rearrange(x, "b c h w -> b (h w) c")

        return x, h, w


class PatchUnembedding(torch.nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, patch_size: int) -> None:
        super().__init__()

        self.conv = zero_init(torch.nn.ConvTranspose2d(hidden_dimension, input_dimension, patch_size, patch_size, padding=0, bias=False))

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.conv(x)

        return x


class Modulation(torch.nn.Module):
    def __init__(self, hidden_dimension: int, time_dimension: int) -> None:
        super().__init__()

        self.linear = zero_init(torch.nn.Linear(time_dimension, hidden_dimension * 2, bias=True))

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(torch.nn.functional.silu(time)).chunk(2, dim=-1)
        x = modulate(x, shift, scale)

        return x


class GatedModulation(torch.nn.Module):
    def __init__(self, hidden_dimension: int, time_dimension: int) -> None:
        super().__init__()

        self.linear = zero_init(torch.nn.Linear(time_dimension, hidden_dimension * 3, bias=True))

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.linear(torch.nn.functional.silu(time)).chunk(3, dim=-1)
        x = modulate(x, shift, scale)

        return x, gate


class Attention(torch.nn.Module):
    def __init__(self, hidden_dimension: int, time_dimension: int, heads: int, eps: float) -> None:
        super().__init__()

        self.heads = heads
        self.norm_1 = torch.nn.LayerNorm(hidden_dimension, eps=eps, elementwise_affine=False)
        self.norm_2 = torch.nn.LayerNorm(hidden_dimension // heads, eps=eps, bias=False)
        self.norm_3 = torch.nn.LayerNorm(hidden_dimension // heads, eps=eps, bias=False)
        self.modulation = GatedModulation(hidden_dimension, time_dimension)
        self.linear_1 = torch.nn.Linear(hidden_dimension, hidden_dimension * 3, bias=False)
        self.linear_2 = torch.nn.Linear(hidden_dimension, hidden_dimension, bias=False)

    def forward(self, x: torch.Tensor, time: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        x, gate = self.modulation(self.norm_1(x), time)
        q, k, v = einops.rearrange(self.linear_1(x), "b t (n h e) -> n b h t e", n=3, h=self.heads)
        q = apply_rope(self.norm_2(q), position)
        k = apply_rope(self.norm_3(k), position)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = self.linear_2(einops.rearrange(x, "b h t e -> b t (h e)")) * gate

        return x


class MLP(torch.nn.Module):
    def __init__(self, hidden_dimension: int, time_dimension: int, eps: float) -> None:
        super().__init__()

        self.norm = torch.nn.LayerNorm(hidden_dimension, elementwise_affine=False, eps=eps)
        self.modulation = GatedModulation(hidden_dimension, time_dimension)
        self.linear_1 = torch.nn.Linear(hidden_dimension, hidden_dimension * 3, bias=False)
        self.linear_2 = torch.nn.Linear(hidden_dimension, hidden_dimension * 3, bias=False)
        self.linear_3 = torch.nn.Linear(hidden_dimension * 3, hidden_dimension, bias=False)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        x, gate = self.modulation(self.norm(x), time)
        x = self.linear_1(x) * torch.nn.functional.silu(self.linear_2(x))
        x = self.linear_3(x) * gate

        return x


class DiTBlock(torch.nn.Module):
    def __init__(self, config: DiTConfig) -> None:
        super().__init__()

        self.attention = Attention(config.hidden_dimension, config.time_dimension, config.heads, config.eps)
        self.mlp = MLP(config.hidden_dimension, config.time_dimension, config.eps)

    def forward(self, x: torch.Tensor, time: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x, time, position)
        x = x + self.mlp(x, time)

        return x


class DiT(torch.nn.Module):
    def __init__(self, config: DiTConfig) -> None:
        super().__init__()

        self.config = config
        self.norm = torch.nn.LayerNorm(config.hidden_dimension, eps=config.eps, elementwise_affine=False, bias=False)
        self.modulation = Modulation(config.hidden_dimension, config.time_dimension)
        self.time_embedding = TimeEmbedding(config.time_dimension)
        self.patch_embedding = PatchEmbedding(config.input_dimension, config.hidden_dimension, config.patch_size)
        self.patch_unembedding = PatchUnembedding(config.input_dimension, config.hidden_dimension, config.patch_size)
        self.blocks = torch.nn.ModuleList([DiTBlock(config) for _ in range(config.layers)])

        self.register_buffer(
            "position",
            create_position(
                config.hidden_dimension,
                config.heads,
                config.sequence_length,
            ),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time = self.time_embedding(time)
        x, h, w = self.patch_embedding(x)

        for block in self.blocks:
            x = block(x, time, self.position)

        x = self.norm(x)
        x = self.modulation(x, time)
        x = self.patch_unembedding(x, h, w)

        return x
