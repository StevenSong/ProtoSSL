from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class PretrainedMixin(ABC, nn.Module):

    @property
    @abstractmethod
    def allow_extra_keys(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def allow_missing_keys(self) -> list[str]:
        pass

    @property
    def allow_size_mismatched_keys(self) -> list[str]:
        return []

    def load_pretrained_weights(self, pretrained_weights: str):
        print(f"==============load_pretrained_weights==============")
        sd = torch.load(
            pretrained_weights,
            weights_only=False,
            map_location="cpu",
        )

        # unwrap if pretrained_weights is a ckpt from a LightningModule
        prefix = ""
        if "state_dict" in sd:
            sd = sd["state_dict"]
            prefix = "model."

        sd = {k.removeprefix(prefix): v for k, v in sd.items()}
        for k in self.allow_size_mismatched_keys:
            v_ckpt = sd.get(k)
            try:
                v_self = self.get_parameter(k)
            except AttributeError as e:
                v_self = None

            if v_ckpt is not None and v_self is not None:
                if v_ckpt.shape != v_self.shape:
                    print(
                        f"Found key {k} in checkpoint has shape {v_ckpt.shape} but differs from expected shape in current model {v_self.shape}."
                        f"Will not load this key from the checkpoint as it is in the whitelist of keys with allowable size mismatches."
                    )
                    del sd[k]

        bad_keys = self.load_state_dict(sd, strict=False)
        bad_extra = set(bad_keys.unexpected_keys) - set(self.allow_extra_keys)
        bad_missing = set(bad_keys.missing_keys) - set(self.allow_missing_keys)
        skipped_extra = set(bad_keys.unexpected_keys) & set(self.allow_extra_keys)
        skipped_missing = set(bad_keys.missing_keys) & set(self.allow_missing_keys)
        if len(skipped_extra) != 0:
            print(
                f"These extra keys were in pretrained_weights but are allowed to be extra according to the current model configuration:\n"
                f"Allowed extra keys: {sorted(list(skipped_extra))}"
            )
        if len(skipped_missing) != 0:
            print(
                f"These keys were missing from pretrained_weights but are allowed to be missing according to the current model configuration:\n"
                f"Allowed missing keys: {sorted(list(skipped_missing))}"
            )
        if len(bad_extra) != 0 or len(bad_missing) != 0:
            raise ValueError(
                f"Tried to load weights from {pretrained_weights} but got unexpected mismatching keys:\n"
                f"Extra keys: {sorted(list(bad_extra))}\n"
                f"Missing keys: {sorted(list(bad_extra))}"
            )
        print(f"Pretrained weights loaded from {pretrained_weights}")
        print(f"===================================================")
