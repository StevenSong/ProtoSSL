import torch

from ..defines import CONV_T, PROT_T, RESNET_T
from ._base_classifier import BaseClassifier
from ._prototype_classifier import PrototypeClassifier
from .encoders import PrototypeEncoderWithAssignment


class PrototypeAssigner(BaseClassifier):
    @property
    def allow_extra_keys(self) -> list[str]:
        return ["proj.weight", "proj.bias", "log_temperature"]

    @property
    def _allow_missing_keys(self) -> list[str]:
        return ["encoder.assignment_weights"]

    def __init__(
        self,
        *,  # enforce kwargs
        resnet_type: RESNET_T,
        conv_type: CONV_T,
        prototype_type: PROT_T,
        n_prototypes: int,
        n_prototypes_per_label: int,
        n_binary_labels: int,
        pretrained_weights: str | None = None,
        input_channels: int = 12,
        partial_len: int | None = None,
        partial_overlap: float | None = None,
    ):
        self.resnet_type = resnet_type
        self.conv_type = conv_type
        self.prototype_type = prototype_type
        self.n_prototypes = n_prototypes
        self.n_prototypes_per_label = n_prototypes_per_label
        self.n_binary_labels = n_binary_labels
        self.partial_len = partial_len
        self.partial_overlap = partial_overlap
        super().__init__(
            encoder=PrototypeEncoderWithAssignment(
                resnet_type=resnet_type,
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

    def static_losses(self) -> torch.Tensor | None:
        # for each label, the prototype assignment slots should pick separate
        # prototypes, so probability distributions should be orthogonal
        x = self.encoder.get_assignments()  # type: ignore - (L, K, P)
        L, K, _ = x.shape

        # TODO: should we normalize distributions...?
        gram = torch.bmm(x, x.mT)  # (L, K, K)

        # ignore the diagonals
        mask = torch.eye(K, device=x.device)  # (K, K) - diag are 1s
        mask = 1 - mask.expand(L, -1, -1)  # (L, K, K) - diag are 0s

        return (gram * mask).pow(2).mean()

    def convert_to_proto_classifier(self) -> PrototypeClassifier:
        model = PrototypeClassifier(
            resnet_type=self.resnet_type,  # type: ignore
            conv_type=self.conv_type,  # type: ignore
            prototype_type=self.prototype_type,  # type: ignore
            n_prototypes=self.n_prototypes_per_label * self.n_binary_labels,
            n_binary_labels=self.n_binary_labels,
            partial_len=self.partial_len,
            partial_overlap=self.partial_overlap,
        )

        # load weights from current state dict into PrototypeClassifier model
        sd = self.state_dict()
        new_sd = dict()
        for name in model.state_dict().keys():
            if name == "encoder.prototypes":
                encoder: PrototypeEncoderWithAssignment = self.encoder  # type: ignore
                assignments = encoder.get_assignments(hard=True)  # (L, K, P)
                indices = assignments.argmax(dim=-1)  # (L, K)
                indices = indices.view(-1)  # (L * K,) - contiguous k blocks
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
        assert set(missing_keys) == set(["cls.weight", "cls.bias"])

        return model
