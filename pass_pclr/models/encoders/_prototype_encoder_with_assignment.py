import torch
import torch.nn as nn
import torch.nn.functional as F

from ...defines import CONV_T, PROT_T, RESNET_T
from ._prototype_encoder import PrototypeEncoder


class PrototypeEncoderWithAssignment(PrototypeEncoder):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        conv_type: CONV_T,
        prototytpe_type: PROT_T,
        n_prototypes: int,
        n_prototypes_per_label: int,
        n_labels: int,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
    ):
        super().__init__(
            resnet_type=resnet_type,
            conv_type=conv_type,
            prototytpe_type=prototytpe_type,
            n_prototypes=n_prototypes,
            partial_len=partial_len,
            partial_overlap=partial_overlap,
        )
        self.assignment_weights = nn.Parameter(
            torch.randn(n_labels, n_prototypes_per_label, n_prototypes),
            requires_grad=True,
        )
        self.ret_per_label = True
        self.emb_dim = n_prototypes_per_label

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sims = super().forward(x)  # (B, P)

        # from ProtoPool: https://github.com/gmum/ProtoPool/blob/2bd42882282fd309b3b70faa62a73c3c88cddd56/model.py#L148
        # NOTE: `gumbel_scale` here is constant scalar (1000), ProtoPool code does ramp up over some number of epochs
        # TODO: consider enforcing separability of prototype slots
        # TODO: maybe consider hard (one-hot) assignment via `y_hard - y_soft.detach() + y_soft`
        # See: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html#torch-nn-functional-gumbel-softmax
        assignments = F.gumbel_softmax(self.assignment_weights * 1000, dim=-1, tau=0.5)

        # assignments has shape (L, K, P), where:
        # L = n_labels, K = n_prototypes_per_label, P = n_prototypes
        # and can be interpreted as:
        # for every label, for every slot (prototype_per_label), what is the
        # probability that a given prototype belongs to that label-slot?

        # compute weighted prototype assignments for each label-slot
        return torch.einsum("bp,lkp->blk", sims, assignments)  # (B, L, K)
