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
        regularize: bool = True,
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

        self.regularize = regularize
        self._l1_ratio_raw = None
        self._alpha_raw = None
        if regularize:
            # per-task elasticnet regularization
            # regularization inspired by sklearn logreg parameters:
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

            # l1_ratio = sigmoid(l1_ratio_raw)
            # ensures l1_ratio is always [0-1]
            self._l1_ratio_raw = nn.Parameter(
                torch.logit(torch.ones(n_binary_labels) * 0.15),
                requires_grad=False,  # fixed regularization parameters
            )
            # alpha = exp(alpha_raw)
            # ensures alpha is always positive
            self._alpha_raw = nn.Parameter(
                torch.log(torch.ones(n_binary_labels) * 1e-4),
                requires_grad=False,  # fixed regularization parameters
            )

        # assumes no other submodules are initialized in subclasses
        if pretrained_weights is not None:
            self.load_pretrained_weights(pretrained_weights)

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

        # per-task elasticnet regularization
        if self.regularize:
            weights = self.cls.weight  # (L * 2, H) - contiguous blocks of (2, H)
            weights = weights.view(-1, 2, weights.shape[-1])  # (L, 2, H) - infer L
            l1 = weights.norm(p=1, dim=(1, 2))  # (L,)
            l2 = weights.norm(p=2, dim=(1, 2))  # (L,)
            l1_ratio = self.l1_ratio
            penalty = self.alpha * (l1_ratio * l1 + (1 - l1_ratio) * (l2**2))  # (L,)
            losses = losses + penalty  # (L,)

        return losses, probs

    # TODO: consider refactoring this method which makes the model intrinsically
    # tied to our trainer, maybe the return of losses above should be a dict?
    def static_losses(self) -> dict[str, torch.Tensor] | None:
        # because the classifier returns a per-task loss, if a subclass has loss
        # components which do not depend on the inputs, they should implement this
        # method which will be called after the main input-dependent forward
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
        return ["cls.*", "_alpha_raw", "_l1_ratio_raw"] + self._allow_missing_keys

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
