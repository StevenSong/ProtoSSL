import torch
import torch.nn as nn


class MultiInputLinear(nn.Module):
    """
    MultiInputLinear creates a separate Linear layer for stacked multi-inputs per sample.

    Consider a `nn.Linear(h_dim, out_dim)` where we want contiguous segments of
    the `out_dim` to be the output of multiple, separate inputs for a single sample,
    i.e. the input is shape `(..., num_inputs, h_dim)` where `out_dim % num_inputs = 0`
    and separate model weights are applied to each input along the `num_inputs` dim.

    This becomes useful as an abstraction of a multi-head classifier that takes
    as task-specific input embeddings. See the BaseClassifier for details.
    """

    def __init__(self, num_inputs: int, *linear_args, **linear_kwargs):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(*linear_args, **linear_kwargs) for _ in range(num_inputs)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input must be shape (..., num_inputs, h_dim)
        num_inputs = len(self.linears)
        if x.shape[-2] != num_inputs:
            raise ValueError(
                f"Provided input has shape {x.shape} but expected size {num_inputs} "
                f"along the -2nd dimension (input must be shape (..., num_inputs, h_dim))"
            )
        outs = []
        for i in range(num_inputs):
            x_i = x[..., i, :]  # (..., h_dim)
            x_i = self.linears[i](x_i)  # (..., out_dim // num_inputs)
            outs.append(x_i)
        return torch.concat(outs, dim=1)  # (..., out_dim)

    def init_weights(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        """
        Initialize weights (and optionally biases) from contiguous blocks.

        Args:
            weight: Tensor of shape (out_features * num_inputs, in_features)
            bias: Optional tensor of shape (out_features * num_inputs,)
        """
        expected_weight_shape = self.weight.shape
        if weight.shape != expected_weight_shape:
            raise ValueError(
                f"Expected weight tensor of shape {expected_weight_shape}, got {weight.shape}"
            )

        if bias is not None:
            expected_bias_shape = (self.weight.shape[0],)
            if bias.shape != expected_bias_shape:
                raise ValueError(
                    f"Expected bias tensor of shape {expected_bias_shape}, got {bias.shape}"
                )

        out_features: int = self.linears[0].out_features  # type: ignore
        with torch.no_grad():
            for i, linear in enumerate(self.linears):
                per_input_weight = weight[i * out_features : (i + 1) * out_features, :]
                linear.weight.copy_(per_input_weight)  # type: ignore
                if bias is not None:
                    if linear.bias is None:
                        raise ValueError(
                            f"Bias provided but linear layer {i} was initialized with bias=False"
                        )
                    per_input_bias = bias[i * out_features : (i + 1) * out_features]
                    linear.bias.copy_(per_input_bias)  # type: ignore

    @property
    def weight(self) -> torch.Tensor:
        return torch.concat([l.weight for l in self.linears], dim=0)  # type: ignore

    @property
    def bias(self) -> torch.Tensor | None:
        if self.linears[0].bias is None:
            return None
        return torch.concat([l.bias for l in self.linears], dim=0)  # type: ignore
