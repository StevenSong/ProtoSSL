import torch

from ..defines import ASSIGN_T, BACKBONE_T, CONV_T, PROT_T
from ._base_classifier import BaseClassifier
from ._prototype_classifier import PrototypeClassifier
from .encoders import PrototypeEncoder, PrototypeEncoderWithAssignment
from .helpers import (
    PrototypeILPAssignmentResult,
    build_association_matrix,
    solve_assignment_ilp,
)


class PrototypeAssigner(BaseClassifier):
    @property
    def allow_extra_keys(self) -> list[str]:
        return [
            # fmt: off
            # from PrototypeContraster (learn-prototypes)
            "proj.*", "log_temperature",
            # from PrototypeSupervisor (learn-prototypes-supervised)
            # NOTE: PrototypeAssigner's encoder uses per-label embeddings
            # and uses a MultiInputLinear instead of a single multitask head
            "cls.bias", "cls.weight",
            # fmt: on
        ]

    @property
    def _allow_missing_keys(self) -> list[str]:
        return ["encoder.assignment_weights"]

    def __init__(
        self,
        *,  # enforce kwargs
        backbone_type: BACKBONE_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        assignment_strategy: ASSIGN_T = "protopool",
        n_prototypes: int,
        n_prototypes_per_label: int,
        n_binary_labels: int,
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
    ):
        self.backbone_type = backbone_type
        self.conv_type = conv_type
        self.prototype_type = prototype_type
        self.assignment_strategy = assignment_strategy

        self.n_prototypes = n_prototypes
        self.n_prototypes_per_label = n_prototypes_per_label
        self.n_binary_labels = n_binary_labels
        self.input_channels = input_channels
        self.partial_len = partial_len
        self.partial_overlap = partial_overlap
        self.lp_indices = None  # set by solve_linear_assignment when assignment_strategy in ["ilp_effect_size", "ilp_effect_size_multiple_allowed"]

        if assignment_strategy == "protopool":
            encoder_cls = PrototypeEncoderWithAssignment
            extra_kwargs = {
                "n_prototypes_per_label": n_prototypes_per_label,
                "n_labels": n_binary_labels,
            }
        elif assignment_strategy in [
            "ilp_effect_size",
            "ilp_effect_size_multiple_allowed",
        ]:
            encoder_cls = PrototypeEncoder
            extra_kwargs = dict()
        else:
            raise ValueError(
                f"Unknown how to make encoder for assignment_strategy={assignment_strategy}"
            )

        super().__init__(
            encoder=encoder_cls(
                backbone_type=backbone_type,
                n_prototypes=n_prototypes,
                conv_type=conv_type,
                prototytpe_type=prototype_type,
                input_channels=input_channels,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
                **extra_kwargs,
            ),
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
        )

    def static_losses(self) -> dict[str, torch.Tensor] | None:
        # ILP does not train assignment weights, so no static loss
        # we shouldn't ever reach this point anyways but just in case
        if self.assignment_strategy != "protopool":
            print(
                "WARNING: Unexpected call to static_losses in PrototypeAssigner with non-prototpool assignment strategy"
            )
            return None

        # In ProtoPool, for each label, the prototype assignment slots should pick
        # separate prototypes, so probability distributions should be orthogonal
        x = self.encoder.get_assignments(hard=True)  # type: ignore - (L, K, P)
        L, K, _ = x.shape

        gram = torch.bmm(x, x.mT)  # (L, K, K)

        # zero out diagonal, penalize off-diagonal overlap
        mask = torch.eye(K, device=x.device)
        mask = 1 - mask.expand(L, -1, -1)

        return {"Prototype_Orthogonality": (gram * mask).pow(2).mean()}

    def solve_linear_assignment(
        self, similarities: torch.Tensor, labels: torch.Tensor
    ) -> PrototypeILPAssignmentResult:
        print("==============solve_linear_assignment==============")
        print(
            f"Solving Linear Assignment (assignment_strategy={self.assignment_strategy})"
        )
        assert self.assignment_strategy in [
            "ilp_effect_size",
            "ilp_effect_size_multiple_allowed",
        ]
        max_classes_per_prototype = (
            labels.shape[1]
            if self.assignment_strategy == "ilp_effect_size_multiple_allowed"
            else 1
        )

        association_matrix, valid_class_mask = build_association_matrix(
            similarities.numpy(),
            labels.numpy(),
            n_min=1,  # min positive samples
            trim=0.10,
            eps=1e-6,
            n_neg_repeats=10,  # number of resample repeats
            balanced_negative_sampling=True,
            random_seed=0,
        )

        result = solve_assignment_ilp(
            association_matrix,
            n_prototypes_per_label=self.n_prototypes_per_label,
            max_classes_per_prototype=max_classes_per_prototype,
            valid_class_mask=valid_class_mask,
        )

        self.lp_indices = torch.as_tensor(
            result.selected_indices_by_class, dtype=torch.long
        )

        print("===================================================")
        return result

    def convert_to_proto_classifier(self) -> PrototypeClassifier:
        model = PrototypeClassifier(
            backbone_type=self.backbone_type,  # type: ignore
            conv_type=self.conv_type,  # type: ignore
            prototype_type=self.prototype_type,  # type: ignore
            n_prototypes=self.n_prototypes_per_label * self.n_binary_labels,
            n_binary_labels=self.n_binary_labels,
            input_channels=self.input_channels,
            partial_len=self.partial_len,
            partial_overlap=self.partial_overlap,
        )

        sd = self.state_dict()
        new_sd = dict()

        for name in model.state_dict().keys():
            if name == "encoder.prototypes":
                encoder: PrototypeEncoderWithAssignment = self.encoder  # type: ignore
                indices = self.get_assignment_indices()
                prototypes = encoder.prototypes[
                    indices.to(device=encoder.prototypes.device)
                ]
                new_sd[name] = prototypes
            elif name == "cls.weight" or name == "cls.bias":
                # PrototypeAssigner classifier head takes a similarity vector of size n_prototypes_per_label
                # (the similarities of the selected prototypes for a given label)
                # whereas PrototypeClassifier classifier head takes a similarity vector over all prototypes
                # these are 2 fundamentally incompatible to convert so just skip the task head
                # afterall, the classifiers are not used for projection and everything else is frozen for
                # final classifier training anyways
                continue
            else:
                if name not in sd:
                    raise ValueError(
                        f"Parameter {name} in PrototypeClassifier not found in PrototypeAssigner state dict"
                    )
                new_sd[name] = sd[name]

        missing_keys, unexpected_keys = model.load_state_dict(new_sd, strict=False)
        assert len(unexpected_keys) == 0
        assert set(missing_keys) == {"cls.weight", "cls.bias"}

        return model

    def get_assignment_indices(self) -> torch.Tensor:
        if self.assignment_strategy == "protopool":
            # ProtoPool conversion path:
            # use hard assignments from encoder.assignment_weights
            assert isinstance(self.encoder, PrototypeEncoderWithAssignment)
            assignments = self.encoder.get_assignments(hard=True)  # (L, K, P)
            indices = assignments.argmax(dim=-1)  # (L, K)
            indices = indices.view(-1)  # (L * K,)
        elif self.assignment_strategy in [
            "ilp_effect_size",
            "ilp_effect_size_multiple_allowed",
        ]:
            if self.lp_indices is None:
                raise ValueError(
                    f"lp_indices is None, has solve_linear_assignment been called yet?"
                )
            # ILP conversion path:
            # lp_indices:
            #     Long tensor of shape (L, K), where:
            #         - L = n_binary_labels
            #         - K = n_prototypes_per_label
            #     and indices[c, :] are the selected prototype ids for class c.
            if self.lp_indices.ndim != 2:
                raise ValueError(
                    f"indices must have shape (n_binary_labels, n_prototypes_per_label), got {tuple(self.lp_indices.shape)}"
                )

            expected_shape = (self.n_binary_labels, self.n_prototypes_per_label)
            if tuple(self.lp_indices.shape) != expected_shape:
                raise ValueError(
                    f"indices must have shape {expected_shape}, got {tuple(self.lp_indices.shape)}"
                )

            if self.lp_indices.dtype not in (torch.int32, torch.int64):
                self.lp_indices = self.lp_indices.long()

            indices = self.lp_indices.reshape(-1)

            if torch.any(indices < 0):
                raise ValueError("indices contains negative prototype ids")
            if torch.any(indices >= self.n_prototypes):
                raise ValueError(
                    f"indices contains prototype ids >= n_prototypes ({self.n_prototypes})"
                )

            # Require unique prototype usage to match the ILP formulation.
            unique_flat = torch.unique(indices)
            if (
                self.assignment_strategy == "ilp_effect_size"
                and unique_flat.numel() != indices.numel()
            ):
                raise ValueError(
                    "ilp_effect_size assignment expects unique prototype ids, "
                    "but duplicate indices were provided."
                )
        else:
            raise ValueError(
                f"Unknown how to interpret assignment indices for assignment_strategy={self.assignment_strategy}"
            )
        return indices
