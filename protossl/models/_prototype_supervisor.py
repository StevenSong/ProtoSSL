import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defines import BACKBONE_T, CONV_T, LABEL_T, PROT_T, SIM_MAX
from ._pretrained_utils import PretrainedMixin
from .encoders import PrototypeEncoder


class PrototypeSupervisor(PretrainedMixin, nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes_per_label: int,
        label_type: LABEL_T = "binary-multilabel",
        label_weights: torch.Tensor,
        label_cooccurrence: torch.Tensor,
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        prototype_h: int | None = None,
        prototype_w: int | None = None,
        use_default_weights: bool = False,
    ):
        super().__init__()
        self.n_prototypes_per_label = n_prototypes_per_label
        self.n_labels = label_weights.shape[0]
        self.register_buffer("label_weights", label_weights, persistent=False)
        self.register_buffer("label_cooccurrence", label_cooccurrence, persistent=False)
        self.encoder = PrototypeEncoder(
            backbone_type=backbone_type,
            n_prototypes=self.n_labels * n_prototypes_per_label,
            conv_type=conv_type,
            prototype_type=prototype_type,
            input_channels=input_channels,
            partial_len=partial_len,
            partial_overlap=partial_overlap,
            prototype_h=prototype_h,
            prototype_w=prototype_w,
        )
        if label_type == "binary-multilabel":
            n_outputs_per_label = 2
        elif label_type == "multiclass":
            print(f"===========PrototypeSupervisor.__init__============")
            print(
                f"Using multiclass mode for classifier (instead of binary multilabel)"
            )
            print(f"===================================================")
            n_outputs_per_label = 1
        else:
            raise ValueError(f"Unknown how to handle label_type={label_type}")
        self.label_type: LABEL_T = label_type
        self.cls = nn.Linear(
            in_features=self.encoder.emb_dim,
            out_features=self.n_labels * n_outputs_per_label,
        )

        if use_default_weights:
            print(f"===========PrototypeSupervisor.__init__============")
            print(f"Using default loss weights (and ignoring label co-occurrence term)")
            print(f"===================================================")
            # original ProtoPNet coefficients
            self.lam_clst = 0.8
            self.lam_sep = 0.08
            self.lam_div = 100
            self.lam_cntrst = 0  # no label co-occurrence loss
        else:
            # ProtoECGNet paper
            self.lam_clst = 0.004
            self.lam_sep = 0.0004
            self.lam_div = 250.0
            self.lam_cntrst = 300.0

        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        # from ProtoECGNet: https://arxiv.org/pdf/2504.08713

        sims: torch.Tensor = self.encoder(x)  # (B, P)
        label_weights: torch.Tensor = self.label_weights  # type: ignore
        logits = self.cls(sims)  # (B, [2]L)

        if self.label_type == "binary-multilabel":
            # binary cross entropy loss
            losses = []
            for i in range(self.n_labels):
                i = i * 2
                per_label_logits = logits[:, i : i + 2]  # (B, 2)
                wt = torch.ones(
                    2, dtype=label_weights.dtype, device=label_weights.device
                )
                wt[1] = label_weights[i // 2]  # use weights: [1, pos_weight]
                per_label_loss = F.cross_entropy(
                    input=per_label_logits,  # (B, 2)
                    target=y[:, i // 2],  # (B,)
                    weight=wt,  # (2,)
                )
                losses.append(per_label_loss)
            # NOTE: Eq 3 in ProtoECGNet paper does not show mean reduction over classes but seems like it should be based on their codebase
            cls_loss = torch.stack(losses).mean()
        elif self.label_type == "multiclass":
            cls_loss = F.cross_entropy(logits, y.argmax(dim=-1), weight=label_weights)
        else:
            raise ValueError(f"Unknown forward for label_type={self.label_type}")

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
            "Classification": cls_loss,
            "Clustering": self.lam_clst * clst_loss,
            "Separation": self.lam_sep * sep_loss,
            "Diversity": self.lam_div * div_loss,
            "Contrastive": self.lam_cntrst * cntrst_loss,
        }

    @property
    def allow_extra_keys(self) -> list[str]:
        return ["_alpha_raw", "_l1_ratio_raw"]

    @property
    def allow_missing_keys(self) -> list[str]:
        return []
