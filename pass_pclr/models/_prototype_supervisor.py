import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defines import CONV_T, PROT_T, RESNET_T, SIM_MAX
from ._pretrained_utils import PretrainedMixin
from .encoders import PrototypeEncoder

class PrototypeSupervisor(PretrainedMixin, nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes_per_label: int,
        label_weights: torch.Tensor,
        label_cooccurrence: torch.Tensor,
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        prototype_h: int | None = None,
        prototype_w: int | None = None,
        audio_backbone_name: str | None = None,
    ):
        super().__init__()
        self.n_prototypes_per_label = n_prototypes_per_label
        self.n_binary_labels = label_weights.shape[0]
        self.register_buffer("label_weights", label_weights, persistent=False)
        self.register_buffer("label_cooccurrence", label_cooccurrence, persistent=False)
        self.encoder = PrototypeEncoder(
            resnet_type=resnet_type,
            n_prototypes=self.n_binary_labels * n_prototypes_per_label,
            conv_type=conv_type,
            prototype_type=prototype_type,
            input_channels=input_channels,
            partial_len=partial_len,
            partial_overlap=partial_overlap,
            prototype_h=prototype_h,
            prototype_w=prototype_w,
            audio_backbone_name=audio_backbone_name,
        )
        self.cls = nn.Linear(
            in_features=self.encoder.emb_dim,
            out_features=self.n_binary_labels * 2,
        )

        # these values were taken from ProtoECGNet non-contrastive experiments; same as ProtoPNet, NEJM AI EEG paper, etc.
        self.lam_clst = 0.8
        self.lam_sep = 0.08 
        self.lam_cntrst = 0 #for audio, remove co-occurrence loss
        self.lam_div = 100

        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # from ProtoECGNet: https://arxiv.org/pdf/2504.08713

        sims: torch.Tensor = self.encoder(x)  # (B, P)
        label_weights: torch.Tensor = self.label_weights  # type: ignore

        # binary cross entropy loss
        logits = self.cls(sims)  # (B, 2 * L)
        losses = []
        for i in range(self.n_binary_labels):
            i = i * 2
            per_label_logits = logits[:, i : i + 2]  # (B, 2)
            wt = torch.ones(2, dtype=label_weights.dtype, device=label_weights.device)
            wt[1] = label_weights[i // 2]  # use weights: [1, pos_weight]
            per_label_loss = F.cross_entropy(
                input=per_label_logits,  # (B, 2)
                target=y[:, i // 2],  # (B,)
                weight=wt,  # (2,)
            )
            losses.append(per_label_loss)
        # NOTE: Eq 3 in ProtoECGNet paper does not show mean reduction over classes but seems like it should be based on their codebase
        bce_loss = torch.stack(losses).mean()

        # clustering loss
        # use repeat_interleave as all prototypes for a given label should be contiguous
        pos_mask = y.repeat_interleave(self.n_prototypes_per_label, 1)  # (B, P)
        neg_mask = 1 - pos_mask
        # only consider similarity for prototypes assigned to the sample
        # since we take the max of the valid similarities, mask invalid entries
        # with similarities less than all other similarities
        pos_prot_sims = pos_mask * sims + neg_mask * -SIM_MAX  # (B, P)
        per_sample_max_pos_sim, _ = pos_prot_sims.max(1)  # (B,)
        clst_loss = -per_sample_max_pos_sim.mean()

        # separation loss
        neg_prot_sims = neg_mask * sims + pos_mask * -SIM_MAX  # (B, P)
        per_sample_max_neg_sim, _ = neg_prot_sims.max(1)  # (B,)
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

        #Print statements (remove later) ############################################################

        print(f"BCE loss: {bce_loss}")
        print(f"Clst loss: {self.lam_clst * clst_loss}")
        print(f"Sep loss: {self.lam_sep * sep_loss}")
        print(f"Div loss: {self.lam_div * div_loss}")
        print(f"Cntrst loss: {self.lam_cntrst * cntrst_loss}")

        #############################################################################################

        return (
            bce_loss
            + self.lam_clst * clst_loss
            + self.lam_sep * sep_loss
            + self.lam_div * div_loss
            + self.lam_cntrst * cntrst_loss
        )

    @property
    def allow_extra_keys(self) -> list[str]:
        return []

    @property
    def allow_missing_keys(self) -> list[str]:
        return []
