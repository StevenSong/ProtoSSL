from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ..defines import BACKBONE_T, CONV_T, PROT_T, SIM_MAX, STAGE_T
from ._pretrained_utils import PretrainedMixin
from .encoders import PrototypeEncoder


@dataclass
class JointStageLambdas:
    lam_clst: float
    lam_sep: float
    lam_div: float
    lam_cntrst: float


@dataclass
class ClsStageLambdas:
    lam_l1: float


LOSS_WEIGHT_T = (
    Literal["protopnet", "protoecgnet"] | JointStageLambdas | ClsStageLambdas
)


class ProtoECGNet(PretrainedMixin, nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        pipeline_stage: STAGE_T,
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes_per_label: int,
        label_names: list[str],
        label_weights: torch.Tensor,
        label_cooccurrence: torch.Tensor | None = None,
        pretrained_weights: str | None = None,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
        loss_weights: LOSS_WEIGHT_T = "protoecgnet",
    ):
        super().__init__()
        if (
            pipeline_stage != "learn-prototypes-supervised"
            and pipeline_stage != "train-classifier"
        ):
            raise ValueError(
                f"Invalid pipeline_stage for ProtoECGNet: {pipeline_stage}"
            )
        if pipeline_stage == "train-classifier" and pretrained_weights is None:
            raise ValueError(
                f"Stage train-classifier must be used with pretrained weights"
            )
        if (
            pipeline_stage == "learn-prototypes-supervised"
            and label_cooccurrence is None
        ):
            raise ValueError(
                "Stage learn-prototypes-supervised needs non-None label_cooccurrence"
            )
        self.pipeline_stage = pipeline_stage
        self.n_prototypes_per_label = n_prototypes_per_label
        self.n_labels = len(label_names)
        self.label_names = label_names
        self.register_buffer("label_weights", label_weights, persistent=False)
        self.register_buffer("label_cooccurrence", label_cooccurrence, persistent=False)
        self.encoder = PrototypeEncoder(
            backbone_type=backbone_type,
            n_prototypes=self.n_labels * n_prototypes_per_label,
            conv_type=conv_type,
            prototype_type=prototype_type,
            partial_len=partial_len,
            partial_overlap=partial_overlap,
        )

        # mask pos/neg prototype class connections
        assign = torch.eye(self.n_labels)  # (L, L)
        assign = assign.repeat_interleave(n_prototypes_per_label, dim=1)  # (L, LP)
        off_class_mask = 1.0 - assign  # (L, LP)
        self.register_buffer("off_class_mask", off_class_mask, persistent=False)

        # binary multilabel classification output
        self.cls = nn.Linear(
            in_features=self.encoder.emb_dim,
            out_features=self.n_labels,
            bias=False,
        )
        # initialize with 1 for pos connects and -0.5 for neg connects
        with torch.no_grad():
            self.cls.weight.copy_(assign - 0.5 * off_class_mask)

        if loss_weights == "protopnet":
            # original ProtoPNet coefficients
            self.lam_clst = 0.8
            self.lam_sep = 0.08
            self.lam_div = 100
            self.lam_cntrst = 0  # no label co-occurrence loss
            self.lam_l1 = 1e-4
        elif loss_weights == "protoecgnet":
            # ProtoECGNet paper
            self.lam_clst = 0.004
            self.lam_sep = 0.0004
            self.lam_div = 250.0
            self.lam_cntrst = 300
            self.lam_l1 = 1e-4
        elif (
            isinstance(loss_weights, JointStageLambdas)
            and pipeline_stage == "learn-prototypes-supervised"
        ):
            self.lam_clst = loss_weights.lam_clst
            self.lam_sep = loss_weights.lam_sep
            self.lam_div = loss_weights.lam_div
            self.lam_cntrst = loss_weights.lam_cntrst
        elif (
            isinstance(loss_weights, ClsStageLambdas)
            and pipeline_stage == "train-classifier"
        ):
            self.lam_l1 = loss_weights.lam_l1
        else:
            raise ValueError(
                f"Unknown how to derive prototype loss weights (got {loss_weights}) for stage (got {pipeline_stage})"
            )

        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[
        dict[str, torch.Tensor],  # losses
        dict[str, torch.Tensor],  # probs
    ]:
        sims: torch.Tensor = self.encoder(x)  # (B, P)
        logits: torch.Tensor = self.cls(sims)  # (B, L)
        probs = logits.sigmoid()  # (B, L)
        if self.pipeline_stage == "learn-prototypes-supervised":
            losses = self._joint_criterion(sims, logits, y)
        elif self.pipeline_stage == "train-classifier":
            losses = self._classifier_criterion(sims, logits, y)
        else:
            raise ValueError(
                f"Unknown how to do forward pass for ProtoECGNet during stage {self.pipeline_stage}"
            )
        return losses, {label: probs[:, i] for i, label in enumerate(self.label_names)}

    def _classifier_criterion(
        self,
        sims: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # ProtoECGNet: https://arxiv.org/pdf/2504.08713
        # Appendix E, Stage 3 - NOTE: this is just the comparison to the branch-specific classifier

        cls_loss = F.binary_cross_entropy_with_logits(
            logits,  # (B, L)
            y.float(),  # (B, L)
            pos_weight=self.label_weights,  # type: ignore
            reduction="mean",
        )

        # masked L1 regularization
        penalty = (self.cls.weight.abs() * self.off_class_mask).sum()  # type: ignore

        return {
            "Classification": cls_loss,
            "Penalty": self.lam_l1 * penalty,
        }

    def _joint_criterion(
        self,
        sims: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # ProtoECGNet: https://arxiv.org/pdf/2504.08713
        # Appendix E, Stage 1

        # binary cross entropy loss - no masked l1 penalty
        cls_loss = F.binary_cross_entropy_with_logits(
            logits,  # (B, L)
            y.float(),  # (B, L)
            pos_weight=self.label_weights,  # type: ignore
            reduction="mean",  # NOTE: Eq 3 in ProtoECGNet paper does not show mean reduction over classes but seems like it should be based on their codebase
        )

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
        return []

    @property
    def allow_missing_keys(self) -> list[str]:
        return ["cls.weight"]


@dataclass
class BranchCfg:
    name: str
    config: str  # path to yaml
    pretrained_weights: str | None = None

    # set by post init
    backbone_type: BACKBONE_T = field(init=False)
    conv_type: CONV_T = field(init=False)
    prototype_type: PROT_T = field(init=False)
    n_prototypes_per_label: int = field(init=False)
    label_subset: list[str] = field(init=False)
    partial_len: int | None = field(default=None, init=False)
    partial_overlap: float | None = field(default=None, init=False)

    def __post_init__(self):
        c = OmegaConf.load(self.config)
        OmegaConf.set_struct(c, True)
        self.backbone_type = c.model.init_args.backbone_type
        self.conv_type = c.model.init_args.conv_type
        self.prototype_type = c.model.init_args.prototype_type
        self.n_prototypes_per_label = c.model.init_args.n_prototypes_per_label
        self.label_subset = c.data.init_args.label_subset
        # NOTE: we don't validate correctness, we're just trying to read
        # required/optional values from the config yaml here
        for opt_k in ["partial_len", "partial_overlap"]:
            if opt_k in c.model.init_args:
                v = c.model.init_args.get(opt_k)
                setattr(self, opt_k, v)


BRANCHES_T = list[BranchCfg]


class ProtoECGNetFusion(PretrainedMixin, nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        pipeline_stage: STAGE_T,
        label_names: list[str],  # must have same ordering as input y to forward
        label_weights: torch.Tensor,
        branches: BRANCHES_T,
        loss_weights: dict[str, float] | LOSS_WEIGHT_T = "protoecgnet",
        pretrained_weights: str | None = None,
    ):
        super().__init__()
        if pipeline_stage != "train-fusion-classifier":
            raise ValueError(
                f"Invalid pipeline_stage for ProtoECGNetFusion: {pipeline_stage}"
            )
        self.pipeline_stage = pipeline_stage
        self.label_names = label_names
        total_labels = len(label_names)
        self.register_buffer("label_weights", label_weights, persistent=False)
        total_encoder_dim, branch_offset = 0, 0
        encoders, label_to_ppl, label_to_slice = {}, {}, {}
        for branch in branches:
            if pretrained_weights is None and branch.pretrained_weights is None:
                raise ValueError(
                    "Must specify either fusion checkpoint or per-branch checkpoint"
                )
            elif (
                pretrained_weights is not None and branch.pretrained_weights is not None
            ):
                raise ValueError(
                    "Cannot provide both fusion checkpoint and per-branch checkpoint"
                )
            n_branch_labels = len(branch.label_subset)
            branch_ppl = branch.n_prototypes_per_label
            encoder = PrototypeEncoder(
                backbone_type=branch.backbone_type,
                n_prototypes=n_branch_labels * branch_ppl,
                conv_type=branch.conv_type,
                prototype_type=branch.prototype_type,
                partial_len=branch.partial_len,
                partial_overlap=branch.partial_overlap,
            )
            if branch.pretrained_weights is not None:
                # only load encoder weights
                sd = torch.load(
                    branch.pretrained_weights,
                    weights_only=False,
                    map_location="cpu",
                )
                prefix = "encoder."
                if "state_dict" in sd:
                    sd = sd["state_dict"]
                    prefix = "model." + prefix
                sd = {
                    k.removeprefix(prefix): v
                    for k, v in sd.items()
                    if k.startswith(prefix)
                }
                encoder.load_state_dict(sd, strict=True)
            encoders[branch.name] = encoder
            total_encoder_dim += encoder.emb_dim

            # encoder branch --> cls mapping
            for branch_idx, label in enumerate(branch.label_subset):
                label_to_ppl[label] = branch_ppl
                # the column slice in the tensor concatenated from all encoders
                label_to_slice[label] = (
                    branch_offset + branch_idx * branch_ppl,  # start
                    branch_offset + (branch_idx + 1) * branch_ppl,  # end
                )
            branch_offset += n_branch_labels * branch_ppl
        self.encoders = nn.ModuleDict(encoders)

        # ensure consistency with branch labels
        label_count = Counter(label_names)
        label_dupes = [x for x, n in label_count.items() if n > 1]
        if len(label_dupes) > 0:
            raise ValueError(f"label_names contains duplicated labels: {label_dupes}")
        branch_count = Counter([l for branch in branches for l in branch.label_subset])
        branch_dupes = [x for x, n in branch_count.items() if n > 1]
        if len(branch_dupes) > 0:
            raise ValueError(f"branches contain duplicated labels: {branch_dupes}")
        missing = set(label_count.keys()) - set(branch_count.keys())
        if len(missing) > 0:
            raise ValueError(
                f"no branch uses these labels (i.e. these labels are missing): {missing}"
            )
        extra = set(branch_count.keys()) - set(label_count.keys())
        if len(extra) > 0:
            raise ValueError(
                f"some branch uses an unknown label (i.e. these labels are extra): {extra}"
            )
        assert branch_offset == total_encoder_dim  # quick sanity check
        reverse_mask = torch.cat(  # (R,)
            [
                torch.arange(start, end)
                for start, end in (label_to_slice[label] for label in label_names)
            ]
        )
        self.register_buffer("reverse_mask", reverse_mask, persistent=False)

        # mask pos/neg prototype class connections
        # let R be the total number of prototypes across all branches
        assign = torch.zeros(total_labels, total_encoder_dim)  # (L, R)
        offset = 0
        # the order of the `cls` weights is given by the order of `label_names`
        for row, label in enumerate(label_names):
            chunk = label_to_ppl[label]
            assign[row, offset : offset + chunk] = 1
            offset += chunk
        off_class_mask = 1.0 - assign  # (L, P')
        self.register_buffer("off_class_mask", off_class_mask, persistent=False)

        # binary multilabel classification output
        # NOTE: deviations from ProtoECGNet codebase:
        # * fusion classifier has no bias term here
        # * fusion classifier does masked weight init
        self.cls = nn.Linear(
            in_features=total_encoder_dim,
            out_features=total_labels,
            bias=False,
        )
        # initialize with 1 for pos connects and -0.5 for neg connects
        with torch.no_grad():
            self.cls.weight.copy_(assign - 0.5 * off_class_mask)

        if loss_weights == "protopnet":
            # original ProtoPNet coefficients
            self.lam_l1 = 1e-4
        elif loss_weights == "protoecgnet":
            # ProtoECGNet paper
            self.lam_l1 = 1e-4
        elif isinstance(loss_weights, ClsStageLambdas):
            self.lam_l1 = loss_weights.lam_l1
        else:
            raise ValueError(
                f"Unknown how to derive prototype loss weights (got {loss_weights})"
            )

        # this should be a checkpoint for the outer fusion model i.e. all branches and final classifier
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[
        dict[str, torch.Tensor],  # losses
        dict[str, torch.Tensor],  # probs
    ]:
        per_branch_sims = []
        for enc in self.encoders.values():
            per_branch_sims.append(enc(x))
        sims: torch.Tensor = torch.concat(per_branch_sims, dim=1)  # (B, R)
        sims = sims[:, self.reverse_mask]  # type: ignore
        logits: torch.Tensor = self.cls(sims)  # (B, L)
        probs = logits.sigmoid()  # (B, L)

        # ProtoECGNet: https://arxiv.org/pdf/2504.08713
        # Appendix E, Stage 3

        cls_loss = F.binary_cross_entropy_with_logits(
            logits,  # (B, L)
            y.float(),  # (B, L)
            pos_weight=self.label_weights,  # type: ignore
            reduction="mean",
        )

        # masked L1 regularization
        penalty = (self.cls.weight.abs() * self.off_class_mask).sum()  # type: ignore

        losses = {"Classification": cls_loss, "Penalty": self.lam_l1 * penalty}
        labeled_probs = {label: probs[:, i] for i, label in enumerate(self.label_names)}
        return losses, labeled_probs

    @property
    def allow_extra_keys(self) -> list[str]:
        return []

    @property
    def allow_missing_keys(self) -> list[str]:
        return []
