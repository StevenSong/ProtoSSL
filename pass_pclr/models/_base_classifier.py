import torch
import torch.nn as nn
import torch.nn.functional as F

from ._pretrained_utils import PretrainedMixin
from .encoders import BaseEncoder
from .layers import MultiInputLinear


class BaseClassifier(PretrainedMixin, nn.Module):
    """
    Binary Multilabel BaseClassifier abstract class

    Classifiers which inherit from this base class need only call
    `super().__init__()` with a given BaseEncoder and the number of labels.
    Additional logic in the subclass `__init__` method may result in incorrect
    loading of pretrained weights.

    Subclasses must also define the properties `allow_[extra|missing]_keys`
    to enable utilities for loading pretrained weights.

    For example:
    ```
    class Classifier(BaseClassifier):
        @property
        def allow_extra_keys(self) -> list[str]:
            return []

        @property
        def allow_missing_keys(self) -> list[str]:
            return []

        def __init__(
            self,
            encoder_params: Any,
            n_binary_labels: int,
            pretrained_weights: str | None = None,
        ):
            super().__init__(
                encoder=Encoder(encoder_params),
                n_binary_labels=n_binary_labels,
                pretrained_weights=pretrained_weights,
            )
    ```
    """

    def __init__(
        self,
        *,  # enforce kwargs
        encoder: BaseEncoder,
        n_binary_labels: int,
        pretrained_weights: str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        if encoder.ret_per_label:
            # encoder returns a per-label embedding
            self.cls = MultiInputLinear(
                num_inputs=n_binary_labels,
                in_features=self.encoder.emb_dim,
                out_features=2,  # binary label output
            )
        else:
            self.cls = nn.Linear(
                in_features=self.encoder.emb_dim,
                out_features=n_binary_labels * 2,
            )

        # assumes no other submodules are initialized in subclasses
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)  # (B, [L,] H), H = hidden_dim
        logits = self.cls(x)  # (B, 2 * L), L = n_binary_labels

        # compute per-label loss
        losses = []
        probs = []
        for i in range(0, logits.shape[1], 2):
            per_label_logits = logits[:, i : i + 2]  # (B, 2)
            per_label_loss = F.cross_entropy(
                input=per_label_logits,  # (B, 2)
                target=y[:, i // 2],  # (B,)
            )
            losses.append(per_label_loss)
            probs.append(F.softmax(per_label_logits, dim=1)[:, 1])

        losses = torch.stack(losses)  # (L,)
        probs = torch.stack(probs)  # (L, B)

        return losses, probs

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("==================freeze_encoder===================")
        print("Froze classifier encoder, only training classification layer")
        print("===================================================")

    @property
    def allow_size_mismatched_keys(self) -> list[str]:
        # if doing transfer learning from one supervised task to another, shapes may differ
        return ["cls.weight", "cls.bias"]
