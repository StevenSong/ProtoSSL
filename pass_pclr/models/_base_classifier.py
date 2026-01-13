import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import BaseEncoder


class BaseClassifier(nn.Module):
    """
    Binary Multilabel BaseClassifier
    """

    def __init__(
        self,
        *,  # enforce kwargs
        encoder: BaseEncoder,
        n_binary_labels: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(
            in_features=self.encoder.emb_dim,
            out_features=n_binary_labels * 2,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)  # (B, H), H = hidden_dim
        logits = self.cls(x)  # (B, 2 * L), L = n_binary_labels

        # compute per-label loss
        losses = []
        probs = []
        for i in range(0, logits.shape[1], 2):
            per_label_logits = logits[:, i : i + 1]  # (B, 2)
            per_label_loss = F.cross_entropy(
                input=per_label_logits,  # (B, 2)
                target=y[:, i // 2],  # (B, 2)
            )
            losses.append(per_label_loss)
            probs.append(F.softmax(per_label_logits, dim=1)[:, 1])

        losses = torch.as_tensor(losses)
        probs = torch.as_tensor(probs)

        return losses, probs

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
