import torch

from ..defines import ASSIGN_T, BACKBONE_T, CONV_T, PROT_T
from ._base_classifier import BaseClassifier
from ._prototype_classifier import PrototypeClassifier
from .encoders import PrototypeEncoderWithAssignment


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

        super().__init__(
            encoder=PrototypeEncoderWithAssignment(
                backbone_type=backbone_type,
                n_prototypes=n_prototypes,
                n_prototypes_per_label=n_prototypes_per_label,
                n_labels=n_binary_labels,
                conv_type=conv_type,
                prototytpe_type=prototype_type,
                input_channels=input_channels,
                partial_len=partial_len,
                partial_overlap=partial_overlap,
            ),
            n_binary_labels=n_binary_labels,
            pretrained_weights=pretrained_weights,
        )

    def static_losses(self) -> dict[str, torch.Tensor] | None:
        # ILP does not train assignment weights, so no static loss
        if self.assignment_strategy != "protopool":
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

    def _build_proto_classifier_shell(self) -> PrototypeClassifier:
        return PrototypeClassifier(
            backbone_type=self.backbone_type,  # type: ignore
            conv_type=self.conv_type,  # type: ignore
            prototype_type=self.prototype_type,  # type: ignore
            n_prototypes=self.n_prototypes_per_label * self.n_binary_labels,
            n_binary_labels=self.n_binary_labels,
            input_channels=self.input_channels,
            partial_len=self.partial_len,
            partial_overlap=self.partial_overlap,
        )

    def convert_to_proto_classifier(self) -> PrototypeClassifier:
        """
        Existing ProtoPool conversion path:
        use hard assignments from encoder.assignment_weights, producing
        contiguous class blocks of size (n_prototypes_per_label).
        """
        model = self._build_proto_classifier_shell()

        sd = self.state_dict()
        new_sd = dict()

        for name in model.state_dict().keys():
            if name == "encoder.prototypes":
                encoder: PrototypeEncoderWithAssignment = self.encoder  # type: ignore[assignment]
                assignments = encoder.get_assignments(hard=True)  # (L, K, P)
                indices = assignments.argmax(dim=-1)  # (L, K)
                indices = indices.view(-1)  # (L * K,)
                prototypes = encoder.prototypes[indices]
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

    def convert_to_proto_classifier_from_indices(
        self,
        indices: torch.Tensor,
    ) -> PrototypeClassifier:
        """
        ILP conversion path.

        Args:
            indices:
                Long tensor of shape (L, K), where:
                  - L = n_binary_labels
                  - K = n_prototypes_per_label
                and indices[c, :] are the selected prototype ids for class c.

        Returns:
            PrototypeClassifier with encoder.prototypes reordered into contiguous
            class-major blocks matching the repo's downstream assumptions.
        """
        if indices.ndim != 2:
            raise ValueError(
                f"indices must have shape (n_binary_labels, n_prototypes_per_label), got {tuple(indices.shape)}"
            )

        expected_shape = (self.n_binary_labels, self.n_prototypes_per_label)
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                f"indices must have shape {expected_shape}, got {tuple(indices.shape)}"
            )

        if indices.dtype not in (torch.int32, torch.int64):
            indices = indices.long()

        flat_indices = indices.reshape(-1)

        if torch.any(flat_indices < 0):
            raise ValueError("indices contains negative prototype ids")
        if torch.any(flat_indices >= self.n_prototypes):
            raise ValueError(
                f"indices contains prototype ids >= n_prototypes ({self.n_prototypes})"
            )

        # Require unique prototype usage to match the ILP formulation.
        unique_flat = torch.unique(flat_indices)
        if unique_flat.numel() != flat_indices.numel():
            raise ValueError(
                "convert_to_proto_classifier_from_indices expects unique prototype ids, "
                "but duplicate indices were provided."
            )

        model = self._build_proto_classifier_shell()

        sd = self.state_dict()
        new_sd = dict()

        for name in model.state_dict().keys():
            if name == "encoder.prototypes":
                encoder: PrototypeEncoderWithAssignment = self.encoder  # type: ignore[assignment]
                prototypes = encoder.prototypes[flat_indices]
                new_sd[name] = prototypes
            elif name == "cls.weight" or name == "cls.bias":
                # Same incompatibility as in ProtoPool conversion.
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
