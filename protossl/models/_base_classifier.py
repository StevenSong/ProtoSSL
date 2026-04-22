from typing import final

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..defines import LABEL_T
from ._pretrained_utils import PretrainedMixin
from .encoders import BaseEncoder
from .layers import MultiInputLinear


class BaseClassifier(PretrainedMixin, nn.Module):
    """
    BaseClassifier abstract class

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
            n_labels: int,
            pretrained_weights: str | None = None,
        ):
            super().__init__(
                encoder=Encoder(encoder_params),
                n_labels=n_labels,
                pretrained_weights=pretrained_weights,
            )
    ```
    """

    def __init__(
        self,
        *,  # enforce kwargs
        encoder: BaseEncoder,
        n_labels: int,
        label_type: LABEL_T = "binary-multilabel",
        pretrained_weights: str | None = None,
        regularize: bool = True,
        regularization_mask: torch.Tensor | None = None,
        l1_ratio_init: float | None = None,
        alpha_init: float | None = None,
        learnable_regularization: bool = False,
    ):
        super().__init__()

        if label_type == "binary-multilabel":
            n_outputs_per_label = 2
        elif label_type == "multiclass":
            print(f"==============BaseClassifier.__init__==============")
            print(
                f"Using multiclass mode for classifier (instead of binary multilabel)"
            )
            print(f"===================================================")
            n_outputs_per_label = 1
        else:
            raise ValueError(f"Unknown how to handle label_type={label_type}")
        self.label_type: LABEL_T = label_type
        self.n_outputs_per_label = n_outputs_per_label

        self.encoder = encoder
        emb_dim = self.encoder.emb_dim
        if encoder.ret_per_label:
            # encoder returns a per-label embedding
            self.cls = MultiInputLinear(
                num_inputs=n_labels,
                in_features=emb_dim,
                out_features=n_outputs_per_label,
            )
        else:
            self.cls = nn.Linear(
                in_features=emb_dim,
                out_features=n_labels * n_outputs_per_label,
            )

        self.regularize = regularize
        if regularization_mask is not None:
            if not regularize:
                raise ValueError(
                    "must set regularize to True if using regularization_mask"
                )
            required_shape = (n_labels, emb_dim)
            if regularization_mask.shape != required_shape:
                raise ValueError(
                    f"regularization_mask must have shape {required_shape} but got shape {regularization_mask.shape}"
                )
            self.register_buffer("regularization_mask", regularization_mask)
        else:
            self.regularization_mask = None
        self._l1_ratio_raw = None
        self._alpha_raw = None
        if regularize:
            # regularization inspired by sklearn logreg parameters:
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
            l1_ratio = l1_ratio_init if l1_ratio_init is not None else 0.15
            alpha = alpha_init if alpha_init is not None else 1e-4

            if label_type == "binary-multilabel":
                # per-task elasticnet regularization

                # l1_ratio = sigmoid(l1_ratio_raw) - ensures l1_ratio is always [0-1]
                self._l1_ratio_raw = nn.Parameter(
                    torch.logit(torch.ones(n_labels) * l1_ratio),
                    requires_grad=learnable_regularization,
                )
                # alpha = exp(alpha_raw) - ensures alpha is always positive
                self._alpha_raw = nn.Parameter(
                    torch.log(torch.ones(n_labels) * alpha),
                    requires_grad=learnable_regularization,
                )
            elif label_type == "multiclass":
                # joint elasticnet regularization
                self._l1_ratio_raw = nn.Parameter(
                    torch.logit(torch.tensor(l1_ratio)),
                    requires_grad=learnable_regularization,
                )
                self._alpha_raw = nn.Parameter(
                    torch.log(torch.tensor(alpha)),
                    requires_grad=learnable_regularization,
                )
            else:
                raise ValueError(f"Unknown regularization for label_type={label_type}")

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

    def _binary_multilabel_forward(
        self,
        logits: torch.Tensor,  # (B, 2L)
        y: torch.Tensor,  # (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            if self.regularization_mask is not None:
                # (L, 2, H)
                mask = self.regularization_mask.unsqueeze(1).expand(-1, 2, -1)
                weights = weights * mask
            l1 = weights.norm(p=1, dim=(1, 2))  # (L,)
            l2 = weights.norm(p=2, dim=(1, 2))  # (L,)
            l1_ratio = self.l1_ratio
            penalty = self.alpha * (l1_ratio * l1 + (1 - l1_ratio) * (l2**2))  # (L,)
            losses = losses + penalty  # (L,)

        return losses, probs

    def _multiclass_forward(
        self,
        logits: torch.Tensor,  # (B, L)
        y: torch.Tensor,  # (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # verify y is one-hot but multiclass
        assert y.ndim == 2, "Expected 2D tensor (B, num_classes)"
        assert (y.sum(dim=-1) == 1).all(), "Rows must sum to 1"
        assert (y >= 0).all() and (y <= 1).all(), "Values must be 0 or 1"

        loss = F.cross_entropy(logits, y.argmax(dim=-1))  # (,) - scalar
        probs = F.softmax(logits, dim=1)  # (B, L)

        # elasticnet regularization
        if self.regularize:
            weights = self.cls.weight  # (L, H)
            if self.regularization_mask is not None:
                weights = weights * self.regularization_mask
            # all scalar values
            l1 = weights.norm(p=1)
            l2 = weights.norm(p=2)
            l1_ratio = self.l1_ratio
            penalty = self.alpha * (l1_ratio * l1 + (1 - l1_ratio) * (l2**2))
            loss = loss + penalty

        return loss, probs

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)  # (B, [L,] H), H = hidden_dim
        logits = self.cls(x)  # (B, [2]L), L = n_labels

        if self.label_type == "binary-multilabel":
            return self._binary_multilabel_forward(logits, y)
        elif self.label_type == "multiclass":
            return self._multiclass_forward(logits, y)
        else:
            raise ValueError(f"Unknown forward for label_type={self.label_type}")

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
