from typing import final

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

    Subclasses must also define the `allow_extra_keys` property and can
    optionally define the `_allow_missing_keys` property (note the leading `_`)
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
        l1_ratio_init: float = 1,
        alpha_init: float = 1e-4,  # disable regularization by setting alpha to 0
        learnable_regularization: bool = False,
        regularization_mask: torch.Tensor | None = None,
        cls_init: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        emb_dim = self.encoder.emb_dim
        if encoder.ret_per_label:
            # encoder returns a per-label embedding
            self.cls = MultiInputLinear(
                num_inputs=n_binary_labels,
                in_features=emb_dim,
                out_features=1,
            )
        else:
            self.cls = nn.Linear(
                in_features=emb_dim,
                out_features=n_binary_labels,
            )

        # selectively control which weights are regularized
        if regularization_mask is not None:
            required_shape = (n_binary_labels, emb_dim)
            if regularization_mask.shape != required_shape:
                raise ValueError(
                    f"regularization_mask must have shape {required_shape} but got shape {regularization_mask.shape}"
                )
            self.register_buffer("regularization_mask", regularization_mask)
        else:
            self.regularization_mask = None

        # elasticnet regularization inspired by sklearn logreg parameters:
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        # l1_ratio = sigmoid(l1_ratio_raw), ensures l1_ratio is always [0-1]
        self._l1_ratio_raw = nn.Parameter(
            torch.logit(torch.as_tensor(l1_ratio_init)),
            requires_grad=learnable_regularization,
        )
        # alpha = exp(alpha_raw), ensures alpha is always positive
        self._alpha_raw = nn.Parameter(
            torch.log(torch.as_tensor(alpha_init)),
            requires_grad=learnable_regularization,
        )

        # assumes no other submodules are initialized in subclasses
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

        # overrides pretrained weights, if any
        if cls_init is not None:
            weight, bias = cls_init
            if isinstance(self.cls, nn.Linear):
                with torch.no_grad():
                    self.cls.weight.copy_(weight)
                    self.cls.bias.copy_(bias)
            else:  # MultiInputLinear
                self.cls.init_weights(weight, bias)

    @property
    def l1_ratio(self) -> torch.Tensor:
        assert self._l1_ratio_raw is not None
        return torch.sigmoid(self._l1_ratio_raw)

    @property
    def alpha(self) -> torch.Tensor:
        assert self._alpha_raw is not None
        return torch.exp(self._alpha_raw)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],  # loss dict
        torch.Tensor,  # probs
    ]:
        embeds = self.encoder(x)  # (B, [L,] H), H = hidden_dim
        logits = self.cls(embeds)  # (B, L), L = n_binary_labels

        bce_loss = F.binary_cross_entropy_with_logits(logits, y.to(torch.float))
        probs = torch.sigmoid(logits)  # (B, L)

        # l1/2 regularization
        weights = self.cls.weight  # (L, H)
        if self.regularization_mask is not None:
            weights = weights * self.regularization_mask
        l1 = weights.norm(p=1)
        l2 = weights.norm(p=2)
        l1_ratio = self.l1_ratio
        penalty = self.alpha * (l1_ratio * l1 + (1 - l1_ratio) * (l2**2))
        bce_loss = bce_loss + penalty

        losses = {"BCE": bce_loss}

        other_losses = self.other_losses(x, y, embeds)
        if other_losses is not None:
            losses |= other_losses
        return losses, probs

    def other_losses(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        # if a subclass has loss components besides the base binary cross entropy,
        # it should implement this method which will be called at the end of the main forward
        return None

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

    @final
    @property
    def allow_missing_keys(self) -> list[str]:
        return [
            "cls.*",
            "_alpha_raw",
            "_l1_ratio_raw",
            "regularization_mask",
        ] + self._allow_missing_keys

    @property
    def _allow_missing_keys(sef) -> list[str]:
        # for base classifier, since there are common whitelisted missing keys
        # that the classifier subclasses should not have to worry about, the
        # subclasses should provide their own specific lists via this property
        return []

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if "allow_missing_keys" in cls.__dict__:
            raise TypeError(
                f"BaseClassifier subclass ({cls.__name__}) should set `_allow_missing_keys` instead of `allow_missing_keys`"
            )
