import torch
import torch.nn.functional as F

from ..defines import BACKBONE_T, CONV_T, PROT_T, SIM_MAX
from ._base_classifier import BaseClassifier
from .encoders import PrototypeEncoder


class PrototypeSupervisor(BaseClassifier):
    def __init__(
        self,
        *,  # enforce kwargs
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes_per_label: int,
        label_weights: torch.Tensor,
        label_cooccurrence: torch.Tensor,
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        l1_ratio_init: float = 1,
        alpha_init: float = 1e-4,  # disable regularization by setting alpha to 0
        learnable_regularization: bool = False,
    ):
        n_binary_labels = label_weights.shape[0]
        mask = torch.repeat_interleave(
            torch.eye(n_binary_labels), n_prototypes_per_label, dim=1
        )
        regularization_mask = 1 - mask

        # apply 1 to prototype connections, -0.5 for others, 0 out bias
        weight = mask + (1 - mask) * -0.5
        bias = torch.zeros(n_binary_labels)
        super().__init__(
            encoder=PrototypeEncoder(
                backbone_type=backbone_type,
                n_prototypes=n_binary_labels * n_prototypes_per_label,
                conv_type=conv_type,
                prototytpe_type=prototype_type,
                input_channels=input_channels,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
            ),
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
            l1_ratio_init=l1_ratio_init,
            alpha_init=alpha_init,
            learnable_regularization=learnable_regularization,
            regularization_mask=regularization_mask,
            cls_init=(weight, bias),
        )
        self.n_prototypes_per_label = n_prototypes_per_label
        self.n_binary_labels = n_binary_labels
        self.register_buffer("label_weights", label_weights, persistent=False)
        self.register_buffer("label_cooccurrence", label_cooccurrence, persistent=False)

        # these values were taken from ProtoECGNet experiments
        self.lam_clst = 0.004
        self.lam_sep = 0.0004
        self.lam_cntrst = 300.0
        self.lam_div = 250.0

        # original ProtoPNet coefficients
        # self.lam_clst = 0.8
        # self.lam_sep = 0.08
        # self.lam_cntrst = 100.0
        # self.lam_div = 0.0

    # called in BaseClassifier's forward function
    def other_losses(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        # ProtoECGNet: https://arxiv.org/pdf/2504.08713
        sims = embeds

        # clustering loss
        # use repeat_interleave as all prototypes for a given label should be contiguous
        pos_mask = y.repeat_interleave(self.n_prototypes_per_label, 1)  # (B, P)
        has_pos = pos_mask.any(dim=1)  # (B,)
        neg_mask = 1 - pos_mask
        has_neg = neg_mask.any(dim=1)  # (B,)
        # only consider similarity for prototypes assigned to the sample
        # since we take the max of the valid similarities, mask invalid entries
        # with similarities less than all other similarities
        pos_prot_sims = pos_mask * sims + neg_mask * -SIM_MAX  # (B, P)
        per_sample_max_pos_sim, _ = pos_prot_sims.max(1)  # (B,)
        per_sample_max_pos_sim = per_sample_max_pos_sim[has_pos]  # (B_p), B_p <= B
        clst_loss = -per_sample_max_pos_sim.mean()

        # separation loss
        neg_prot_sims = neg_mask * sims + pos_mask * -SIM_MAX  # (B, P)
        per_sample_max_neg_sim, _ = neg_prot_sims.max(1)  # (B,)
        per_sample_max_neg_sim = per_sample_max_neg_sim[has_neg]  # (B_n), B_n <= B
        sep_loss = per_sample_max_neg_sim.mean()

        # orthogonality loss
        assert isinstance(self.encoder, PrototypeEncoder)
        prots = F.normalize(self.encoder.prototypes, p=2, dim=1)  # (P, H)
        n_prot = prots.shape[0]
        identity = torch.eye(n_prot, device=prots.device)
        inter_prot_sims = prots @ prots.T  # (P, P)
        # squared Frobenius norm (skip the sqrt) of inter-prototype similarities (except self similarity)
        # NOTE: division by n_prot^2 not in ProtoECGNet paper but in their codebase
        div_loss = ((inter_prot_sims - identity) ** 2).sum() / (n_prot**2)

        # contrastive loss
        cooc: torch.Tensor = self.label_cooccurrence  # type: ignore - (L, L)
        ppl = self.n_prototypes_per_label
        # use Kronecker product to expand cooccurrence matrix to prototype assignemnts
        cooc_kron = torch.kron(cooc, torch.ones(ppl, ppl, device=cooc.device))  # (P, P)
        # NOTE: cooc normalization not in ProtoECGNet paper but in their codebase
        pos_cooc = cooc_kron / (cooc_kron.sum() + 1e-6)
        neg_cooc = (1 - cooc_kron) / ((1 - cooc_kron).sum() + 1e-6)
        pos_weighted_sims = (pos_cooc * inter_prot_sims).sum()
        neg_weighted_sims = (neg_cooc * inter_prot_sims).sum()
        cntrst_loss = (pos_weighted_sims - neg_weighted_sims) / n_prot**0.5

        return {
            "Clustering": self.lam_clst * clst_loss,
            "Separation": self.lam_sep * sep_loss,
            "Diversity": self.lam_div * div_loss,
            "Contrastive": self.lam_cntrst * cntrst_loss,
        }

    @property
    def allow_extra_keys(self) -> list[str]:
        return []

    @property
    def _allow_missing_keys(self) -> list[str]:
        return []
