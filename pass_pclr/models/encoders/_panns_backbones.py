# Backbones below adapted from: https://github.com/qiuqiangkong/audioset_tagging_cnn/

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


def do_mixup(x, mixup_lambda):
    out = (
        x.transpose(0, -1) * mixup_lambda
        + torch.flip(x, dims=[0]).transpose(0, -1) * (1 - mixup_lambda)
    ).transpose(0, -1)
    return out


def interpolate(x, ratio):
    """
    x: (B, T, C)
    returns: (B, T * ratio, C)
    """
    batch_size, time_steps, classes_num = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """
    framewise_output: (B, T, C)
    pad or trim to frames_num along T
    """
    if framewise_output.shape[1] < frames_num:
        pad = frames_num - framewise_output.shape[1]
        last_frame = framewise_output[:, -1:, :].repeat(1, pad, 1)
        framewise_output = torch.cat((framewise_output, last_frame), dim=1)
    else:
        framewise_output = framewise_output[:, :frames_num, :]
    return framewise_output


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def _ensure_waveform_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Accept either:
      (B, T)
      (B, 1, T)
    Return:
      (B, T)
    """
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[1] == 1:
        return x.squeeze(1)
    raise ValueError(f"Expected waveform shape (B,T) or (B,1,T), got {tuple(x.shape)}")


def _ensure_waveform_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Accept either:
      (B, T)
      (B, 1, T)
    Return:
      (B, 1, T)
    """
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3 and x.shape[1] == 1:
        return x
    raise ValueError(f"Expected waveform shape (B,T) or (B,1,T), got {tuple(x.shape)}")


# Geometry-preserving schedules for the FINAL latent map of 2D spectrogram backbones.
# With 64 mel bins:
#   six-stage: 64 -> 32 -> 16 -> 8 -> 8 -> 8 -> 8
#   four-stage: 64 -> 32 -> 16 -> 8 -> 8
# FINAL_MAP_DOWNSAMPLE_6 = [
#     (2, 2),
#     (2, 2),
#     (2, 2),
#     (1, 2),
#     (1, 2),
#     (1, 1),
# ]
FINAL_MAP_DOWNSAMPLE_6 = [
    (2, 2),
    (2, 2),
    (2, 1),
    (2, 1),
    (2, 1),
    (1, 1),
]

FINAL_MAP_DOWNSAMPLE_4 = [
    (2, 2),
    (2, 2),
    (2, 2),
    (2, 1),
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))

        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))

        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation="linear", temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=n_in,
            out_channels=n_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"Unsupported activation {self.activation}")


class _LogmelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        *,
        use_specaug=True,
        time_mixup=False,
        freq_drop_width=8,
    ):
        super().__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.use_specaug = use_specaug
        self.time_mixup = time_mixup
        self.mel_bins = mel_bins

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=freq_drop_width,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)
        init_bn(self.bn0)

    def forward(self, input, mixup_lambda=None):
        x = _ensure_waveform_2d(input)

        if self.time_mixup and self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.use_specaug and self.training:
            x = self.spec_augmenter(x)

        if (not self.time_mixup) and self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        return x  # (B, 1, time, mel)


class _SpectrogramModelMixin:
    """
    Mixin for 2D spectrogram models.

    Convention:
      - internal latent maps use original PANNs-style layout: (B, C, time, mel)
      - local_feature_map(...) returns standardized layout: (B, C, freq, time)
    """

    def local_feature_map(self, input, mixup_lambda=None):
        x = self._forward_latent_map(input, mixup_lambda=mixup_lambda)  # (B,C,T,F)
        return x.permute(0, 1, 3, 2).contiguous()  # (B,C,F,T)


class _SixStageSpectrogramModel(nn.Module, _SpectrogramModelMixin):
    def __init__(
        self,
        *,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        classes_num,
        use_specaug=True,
        use_intermediate_dropout=True,
        time_mixup=False,
        freq_drop_width=8,
        embed_dim=2048,
    ):
        super().__init__()

        self.frontend = _LogmelFrontend(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            use_specaug=use_specaug,
            time_mixup=time_mixup,
            freq_drop_width=freq_drop_width,
        )

        self.spectrogram_extractor = self.frontend.spectrogram_extractor
        self.logmel_extractor = self.frontend.logmel_extractor
        self.spec_augmenter = self.frontend.spec_augmenter
        self.bn0 = self.frontend.bn0

        self.use_intermediate_dropout = use_intermediate_dropout
        self.final_map_downsample = FINAL_MAP_DOWNSAMPLE_6

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, embed_dim, bias=True)
        self.fc_audioset = nn.Linear(embed_dim, classes_num, bias=True)

        self.emb_dim = embed_dim
        self.final_map_channels = 2048

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _drop(self, x):
        if self.use_intermediate_dropout:
            return F.dropout(x, p=0.2, training=self.training)
        return x

    def _conv_backbone(self, x):
        p1, p2, p3, p4, p5, p6 = self.final_map_downsample

        x = self.conv_block1(x, pool_size=p1, pool_type="avg")
        x = self._drop(x)
        x = self.conv_block2(x, pool_size=p2, pool_type="avg")
        x = self._drop(x)
        x = self.conv_block3(x, pool_size=p3, pool_type="avg")
        x = self._drop(x)
        x = self.conv_block4(x, pool_size=p4, pool_type="avg")
        x = self._drop(x)
        x = self.conv_block5(x, pool_size=p5, pool_type="avg")
        x = self._drop(x)
        x = self.conv_block6(x, pool_size=p6, pool_type="avg")
        x = self._drop(x)
        return x  # (B,C,T,F)

    def _forward_latent_map(self, input, mixup_lambda=None):
        x = self.frontend(input, mixup_lambda=mixup_lambda)
        x = self._conv_backbone(x)
        return x

    def forward(self, input, mixup_lambda=None):
        x = self._forward_latent_map(input, mixup_lambda=mixup_lambda)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class _FourStageSpectrogramModel(nn.Module, _SpectrogramModelMixin):
    def __init__(
        self,
        *,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        classes_num,
        block_cls,
        channels,
        embed_dim,
        use_specaug=True,
        freq_drop_width=8,
    ):
        super().__init__()

        self.frontend = _LogmelFrontend(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            use_specaug=use_specaug,
            freq_drop_width=freq_drop_width,
        )

        self.spectrogram_extractor = self.frontend.spectrogram_extractor
        self.logmel_extractor = self.frontend.logmel_extractor
        self.spec_augmenter = self.frontend.spec_augmenter
        self.bn0 = self.frontend.bn0

        self.final_map_downsample = FINAL_MAP_DOWNSAMPLE_4

        self.conv_block1 = block_cls(in_channels=1, out_channels=channels[0])
        self.conv_block2 = block_cls(in_channels=channels[0], out_channels=channels[1])
        self.conv_block3 = block_cls(in_channels=channels[1], out_channels=channels[2])
        self.conv_block4 = block_cls(in_channels=channels[2], out_channels=channels[3])

        self.fc1 = nn.Linear(channels[3], embed_dim, bias=True)
        self.fc_audioset = nn.Linear(embed_dim, classes_num, bias=True)

        self.emb_dim = embed_dim
        self.final_map_channels = channels[3]

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _conv_backbone(self, x):
        p1, p2, p3, p4 = self.final_map_downsample

        x = self.conv_block1(x, pool_size=p1, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=p2, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=p3, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=p4, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        return x  # (B,C,T,F)

    def _forward_latent_map(self, input, mixup_lambda=None):
        x = self.frontend(input, mixup_lambda=mixup_lambda)
        x = self._conv_backbone(x)
        return x

    def forward(self, input, mixup_lambda=None):
        x = self._forward_latent_map(input, mixup_lambda=mixup_lambda)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class Cnn14(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=2048,
        )


class Cnn14_no_specaug(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=2048,
            use_specaug=False,
        )


class Cnn14_no_dropout(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=2048,
            use_intermediate_dropout=False,
        )


class Cnn6(_FourStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            block_cls=ConvBlock5x5,
            channels=[64, 128, 256, 512],
            embed_dim=512,
        )


class Cnn10(_FourStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            block_cls=ConvBlock,
            channels=[64, 128, 256, 512],
            embed_dim=512,
        )


def _resnet_conv3x3(in_planes, out_planes):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        bias=False,
        dilation=1,
    )


def _resnet_conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def _is_stride_identity(stride):
    return stride == 1 or stride == (1, 1)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("_ResnetBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if _is_stride_identity(self.stride):
            out = x
        else:
            out = F.avg_pool2d(x, kernel_size=self.stride)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups
        self.stride = stride

        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if not _is_stride_identity(self.stride):
            x = F.avg_pool2d(x, kernel_size=self.stride)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple"
            )

        self.groups = groups
        self.base_width = width_per_group

        # Preserve final frequency geometry:
        # initial 64 -> 32 from conv_block1
        # layer2: 32 -> 16
        # layer3: 16 -> 8
        # layer4: 8 -> 8
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=(2, 2), dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=(2, 2), dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=(1, 2), dilate=replace_stride_with_dilation[2]
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride if isinstance(stride, int) else stride[-1]
            stride = 1

        if (not _is_stride_identity(stride)) or self.inplanes != planes * block.expansion:
            if _is_stride_identity(stride):
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            else:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride),
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class _ResNetAudioBase(nn.Module, _SpectrogramModelMixin):
    def __init__(
        self,
        *,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        classes_num,
        block,
        layers,
        conv_after_in_channels,
    ):
        super().__init__()

        self.frontend = _LogmelFrontend(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
        )

        self.spectrogram_extractor = self.frontend.spectrogram_extractor
        self.logmel_extractor = self.frontend.logmel_extractor
        self.spec_augmenter = self.frontend.spec_augmenter
        self.bn0 = self.frontend.bn0

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.resnet = _ResNet(block=block, layers=layers, zero_init_residual=True)
        self.conv_block_after1 = ConvBlock(
            in_channels=conv_after_in_channels,
            out_channels=2048,
        )

        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.emb_dim = 2048
        self.final_map_channels = 2048

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _forward_latent_map(self, input, mixup_lambda=None):
        x = self.frontend(input, mixup_lambda=mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training, inplace=False)

        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(1, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=False)

        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training, inplace=False)

        return x  # (B,C,T,F)

    def forward(self, input, mixup_lambda=None):
        x = self._forward_latent_map(input, mixup_lambda=mixup_lambda)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class ResNet22(_ResNetAudioBase):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            block=_ResnetBasicBlock,
            layers=[2, 2, 2, 2],
            conv_after_in_channels=512,
        )


class ResNet38(_ResNetAudioBase):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            block=_ResnetBasicBlock,
            layers=[3, 4, 6, 3],
            conv_after_in_channels=512,
        )


class ResNet54(_ResNetAudioBase):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            block=_ResnetBottleneck,
            layers=[3, 4, 6, 3],
            conv_after_in_channels=2048,
        )


class Cnn14_emb512(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=512,
        )


class Cnn14_emb128(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=128,
        )


class Cnn14_emb32(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=32,
        )


class MobileNetV1(nn.Module, _SpectrogramModelMixin):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.frontend = _LogmelFrontend(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
        )
        self.spectrogram_extractor = self.frontend.spectrogram_extractor
        self.logmel_extractor = self.frontend.logmel_extractor
        self.spec_augmenter = self.frontend.spec_augmenter
        self.bn0 = self.frontend.bn0

        def conv_bn(inp, oup, stride):
            pool = (stride, stride) if isinstance(stride, int) else stride
            layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(pool),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            ]
            layers = nn.Sequential(*layers)
            init_layer(layers[0])
            init_bn(layers[2])
            return layers

        def conv_dw(inp, oup, stride):
            pool = (stride, stride) if isinstance(stride, int) else stride
            layers = [
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.AvgPool2d(pool),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            ]
            layers = nn.Sequential(*layers)
            init_layer(layers[0])
            init_bn(layers[2])
            init_layer(layers[4])
            init_bn(layers[5])
            return layers

        self.features = nn.Sequential(
            conv_bn(1, 32, (2, 2)),
            conv_dw(32, 64, (1, 1)),
            conv_dw(64, 128, (2, 2)),
            conv_dw(128, 128, (1, 1)),
            conv_dw(128, 256, (2, 2)),
            conv_dw(256, 256, (1, 1)),
            conv_dw(256, 512, (1, 2)),
            conv_dw(512, 512, (1, 1)),
            conv_dw(512, 512, (1, 1)),
            conv_dw(512, 512, (1, 1)),
            conv_dw(512, 512, (1, 1)),
            conv_dw(512, 512, (1, 1)),
            conv_dw(512, 1024, (1, 2)),
            conv_dw(1024, 1024, (1, 1)),
        )

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.emb_dim = 1024
        self.final_map_channels = 1024

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _forward_latent_map(self, input, mixup_lambda=None):
        x = self.frontend(input, mixup_lambda=mixup_lambda)
        x = self.features(x)
        return x  # (B,C,T,F)

    def forward(self, input, mixup_lambda=None):
        x = self._forward_latent_map(input, mixup_lambda=mixup_lambda)
        x = torch.mean(x, dim=3)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        stride_id = stride == 1 or stride == (1, 1)

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = stride_id and inp == oup

        pool = (stride, stride) if isinstance(stride, int) else stride

        if expand_ratio == 1:
            layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(pool),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        else:
            layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.AvgPool2d(pool),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]

        layers = nn.Sequential(*layers)

        for m in layers:
            if isinstance(m, nn.Conv2d):
                init_layer(m)
            elif isinstance(m, nn.BatchNorm2d):
                init_bn(m)

        self.conv = layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module, _SpectrogramModelMixin):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.frontend = _LogmelFrontend(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
        )
        self.spectrogram_extractor = self.frontend.spectrogram_extractor
        self.logmel_extractor = self.frontend.logmel_extractor
        self.spec_augmenter = self.frontend.spec_augmenter
        self.bn0 = self.frontend.bn0

        width_mult = 1.0
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        interverted_residual_setting = [
            [1, 16, 1, (1, 1)],
            [6, 24, 2, (2, 2)],
            [6, 32, 3, (2, 2)],
            [6, 64, 4, (1, 2)],
            [6, 96, 3, (1, 2)],
            [6, 160, 3, (1, 1)],
            [6, 320, 1, (1, 1)],
        ]

        def conv_bn(inp, oup, stride):
            pool = (stride, stride) if isinstance(stride, int) else stride
            layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(pool),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            ]
            layers = nn.Sequential(*layers)
            init_layer(layers[0])
            init_bn(layers[2])
            return layers

        def conv_1x1_bn(inp, oup):
            layers = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
            init_layer(layers[0])
            init_bn(layers[1])
            return layers

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, (2, 2))]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.fc1 = nn.Linear(1280, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.emb_dim = 1024
        self.final_map_channels = 1280

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _forward_latent_map(self, input, mixup_lambda=None):
        x = self.frontend(input, mixup_lambda=mixup_lambda)
        x = self.features(x)
        return x  # (B,C,T,F)

    def forward(self, input, mixup_lambda=None):
        x = self._forward_latent_map(input, mixup_lambda=mixup_lambda)
        x = torch.mean(x, dim=3)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class LeeNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, x, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


class LeeNet11(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.conv_block1 = LeeNetConvBlock(1, 64, 3, 3)
        self.conv_block2 = LeeNetConvBlock(64, 64, 3, 1)
        self.conv_block3 = LeeNetConvBlock(64, 64, 3, 1)
        self.conv_block4 = LeeNetConvBlock(64, 128, 3, 1)
        self.conv_block5 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block6 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block7 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block8 = LeeNetConvBlock(128, 128, 3, 1)
        self.conv_block9 = LeeNetConvBlock(128, 256, 3, 1)

        self.fc1 = nn.Linear(256, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.emb_dim = 512

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        x = _ensure_waveform_3d(input)

        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x)
        x = self.conv_block2(x, pool_size=3)
        x = self.conv_block3(x, pool_size=3)
        x = self.conv_block4(x, pool_size=3)
        x = self.conv_block5(x, pool_size=3)
        x = self.conv_block6(x, pool_size=3)
        x = self.conv_block7(x, pool_size=3)
        x = self.conv_block8(x, pool_size=3)
        x = self.conv_block9(x, pool_size=3)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class LeeNetConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)
        return x


class LeeNet24(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.conv_block1 = LeeNetConvBlock2(1, 64, 3, 3)
        self.conv_block2 = LeeNetConvBlock2(64, 96, 3, 1)
        self.conv_block3 = LeeNetConvBlock2(96, 128, 3, 1)
        self.conv_block4 = LeeNetConvBlock2(128, 128, 3, 1)
        self.conv_block5 = LeeNetConvBlock2(128, 256, 3, 1)
        self.conv_block6 = LeeNetConvBlock2(256, 256, 3, 1)
        self.conv_block7 = LeeNetConvBlock2(256, 512, 3, 1)
        self.conv_block8 = LeeNetConvBlock2(512, 512, 3, 1)
        self.conv_block9 = LeeNetConvBlock2(512, 1024, 3, 1)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, classes_num, bias=True)

        self.emb_dim = 1024

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        x = _ensure_waveform_3d(input)

        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block2(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block3(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block4(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block5(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block6(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block7(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block8(x, pool_size=3)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv_block9(x, pool_size=1)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class DaiNetResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        self.conv4 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

        self.downsample = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.bn4 = nn.BatchNorm1d(out_channels)
        self.bn_downsample = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.downsample)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        nn.init.constant_(self.bn4.weight, 0)
        init_bn(self.bn_downsample)

    def forward(self, input, pool_size=1):
        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.relu_(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))

        if input.shape == x.shape:
            x = F.relu_(x + input)
        else:
            x = F.relu(x + self.bn_downsample(self.downsample(input)))

        if pool_size != 1:
            x = F.max_pool1d(x, kernel_size=pool_size, padding=pool_size // 2)

        return x


class DaiNet19(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=80,
            stride=4,
            padding=0,
            bias=False,
        )
        self.bn0 = nn.BatchNorm1d(64)
        self.conv_block1 = DaiNetResBlock(64, 64, 3)
        self.conv_block2 = DaiNetResBlock(64, 128, 3)
        self.conv_block3 = DaiNetResBlock(128, 256, 3)
        self.conv_block4 = DaiNetResBlock(256, 512, 3)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        self.emb_dim = 512

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        x = _ensure_waveform_3d(input)

        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.bn0(self.conv0(x))
        x = self.conv_block1(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.conv_block2(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.conv_block3(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.conv_block4(x)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


def _resnet_conv3x1_wav1d(in_planes, out_planes, dilation):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding=dilation,
        groups=1,
        bias=False,
        dilation=dilation,
    )


def _resnet_conv1x1_wav1d(in_planes, out_planes):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlockWav1d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("_ResnetBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")

        self.stride = stride

        self.conv1 = _resnet_conv3x1_wav1d(inplanes, planes, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x1_wav1d(planes, planes, dilation=2)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride != 1:
            out = F.max_pool1d(x, kernel_size=self.stride)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNetWav1d(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple"
            )

        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=4)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=4)
        self.layer5 = self._make_layer(block, 1024, layers[4], stride=4)
        self.layer6 = self._make_layer(block, 1024, layers[5], stride=4)
        self.layer7 = self._make_layer(block, 2048, layers[6], stride=4)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1_wav1d(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            else:
                downsample = nn.Sequential(
                    nn.AvgPool1d(kernel_size=stride),
                    _resnet_conv1x1_wav1d(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x


class Res1dNet31(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=11,
            stride=5,
            padding=5,
            bias=False,
        )
        self.bn0 = nn.BatchNorm1d(64)
        self.resnet = _ResNetWav1d(_ResnetBasicBlockWav1d, [2, 2, 2, 2, 2, 2, 2])

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.emb_dim = 2048

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        x = _ensure_waveform_3d(input)

        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.bn0(self.conv0(x))
        x = self.resnet(x)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class Res1dNet51(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=11,
            stride=5,
            padding=5,
            bias=False,
        )
        self.bn0 = nn.BatchNorm1d(64)
        self.resnet = _ResNetWav1d(_ResnetBasicBlockWav1d, [2, 3, 4, 6, 4, 3, 2])

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.emb_dim = 2048

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        x = _ensure_waveform_3d(input)

        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.bn0(self.conv0(x))
        x = self.resnet(x)

        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=2,
            padding=2,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        return x


class Wavegram_Cnn14(nn.Module, _SpectrogramModelMixin):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.pre_conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=11,
            stride=5,
            padding=5,
            bias=False,
        )
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.final_map_downsample = FINAL_MAP_DOWNSAMPLE_6[1:]

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.emb_dim = 2048
        self.final_map_channels = 2048

        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _forward_latent_map(self, input, mixup_lambda=None):
        x = _ensure_waveform_3d(input)

        a1 = F.relu_(self.pre_bn0(self.pre_conv0(x)))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        if self.training and mixup_lambda is not None:
            a1 = do_mixup(a1, mixup_lambda)

        p2, p3, p4, p5, p6 = self.final_map_downsample

        x = a1
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=p2, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=p3, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=p4, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=p5, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=p6, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        return x  # (B,C,T,F)

    def forward(self, input, mixup_lambda=None):
        x = self._forward_latent_map(input, mixup_lambda=mixup_lambda)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class Wavegram_Logmel_Cnn14(nn.Module, _SpectrogramModelMixin):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.pre_conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=11,
            stride=5,
            padding=5,
            bias=False,
        )
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        self.frontend = _LogmelFrontend(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
        )

        self.spectrogram_extractor = self.frontend.spectrogram_extractor
        self.logmel_extractor = self.frontend.logmel_extractor
        self.spec_augmenter = self.frontend.spec_augmenter
        self.bn0 = self.frontend.bn0

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.final_map_downsample = FINAL_MAP_DOWNSAMPLE_6

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.emb_dim = 2048
        self.final_map_channels = 2048

        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _forward_latent_map(self, input, mixup_lambda=None):
        wav = _ensure_waveform_3d(input)

        a1 = F.relu_(self.pre_bn0(self.pre_conv0(wav)))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        x = self.frontend(input, mixup_lambda=None)

        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            a1 = do_mixup(a1, mixup_lambda)

        p1, p2, p3, p4, p5, p6 = self.final_map_downsample

        x = self.conv_block1(x, pool_size=p1, pool_type="avg")
        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=p2, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=p3, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=p4, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=p5, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=p6, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        return x  # (B,C,T,F)

    def forward(self, input, mixup_lambda=None):
        x = self._forward_latent_map(input, mixup_lambda=mixup_lambda)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class Wavegram_Logmel128_Cnn14(nn.Module, _SpectrogramModelMixin):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__()

        self.pre_conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=11,
            stride=5,
            padding=5,
            bias=False,
        )
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 256)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        self.frontend = _LogmelFrontend(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            freq_drop_width=16,
        )

        self.spectrogram_extractor = self.frontend.spectrogram_extractor
        self.logmel_extractor = self.frontend.logmel_extractor
        self.spec_augmenter = self.frontend.spec_augmenter
        self.bn0 = self.frontend.bn0

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.final_map_downsample = FINAL_MAP_DOWNSAMPLE_6

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.emb_dim = 2048
        self.final_map_channels = 2048

        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _forward_latent_map(self, input, mixup_lambda=None):
        wav = _ensure_waveform_3d(input)

        a1 = F.relu_(self.pre_bn0(self.pre_conv0(wav)))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 64, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        x = self.frontend(input, mixup_lambda=None)

        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            a1 = do_mixup(a1, mixup_lambda)

        p1, p2, p3, p4, p5, p6 = self.final_map_downsample

        x = self.conv_block1(x, pool_size=p1, pool_type="avg")
        x = torch.cat((x, a1), dim=1)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=p2, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=p3, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=p4, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=p5, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=p6, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        return x  # (B,C,T,F)

    def forward(self, input, mixup_lambda=None):
        x = self._forward_latent_map(input, mixup_lambda=mixup_lambda)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


class Cnn14_16k(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        assert sample_rate == 16000
        assert window_size == 512
        assert hop_size == 160
        assert mel_bins == 64
        assert fmin == 50
        assert fmax == 8000
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=2048,
        )


class Cnn14_8k(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        assert sample_rate == 8000
        assert window_size == 256
        assert hop_size == 80
        assert mel_bins == 64
        assert fmin == 50
        assert fmax == 4000
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=2048,
        )


class Cnn14_mixup_time_domain(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=2048,
            time_mixup=True,
        )


class Cnn14_mel32(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=2048,
            freq_drop_width=4,
        )


class Cnn14_mel128(_SixStageSpectrogramModel):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            embed_dim=2048,
            freq_drop_width=16,
        )


class _Cnn14DecisionLevelBase(nn.Module, _SpectrogramModelMixin):
    def __init__(
        self,
        *,
        sample_rate,
        window_size,
        hop_size,
        mel_bins,
        fmin,
        fmax,
        classes_num,
        mode,
    ):
        super().__init__()
        assert mode in {"max", "avg", "att"}
        self.mode = mode
        self.interpolate_ratio = 32

        self.frontend = _LogmelFrontend(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
        )

        self.spectrogram_extractor = self.frontend.spectrogram_extractor
        self.logmel_extractor = self.frontend.logmel_extractor
        self.spec_augmenter = self.frontend.spec_augmenter
        self.bn0 = self.frontend.bn0

        self.final_map_downsample = FINAL_MAP_DOWNSAMPLE_6

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = None
        self.att_block = None

        self.emb_dim = 2048
        self.final_map_channels = 2048

        if mode in {"max", "avg"}:
            self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
            init_layer(self.fc_audioset)
        else:
            self.att_block = AttBlock(2048, classes_num, activation="sigmoid")

        init_layer(self.fc1)

    def _conv_backbone(self, x):
        p1, p2, p3, p4, p5, p6 = self.final_map_downsample
        x = self.conv_block1(x, pool_size=p1, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=p2, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=p3, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=p4, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=p5, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=p6, pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        return x  # (B,C,T,F)

    def _forward_latent_map(self, input, mixup_lambda=None):
        x = self.frontend(input, mixup_lambda=mixup_lambda)
        x = self._conv_backbone(x)
        return x

    def forward(self, input, mixup_lambda=None):
        x0 = _ensure_waveform_2d(input)
        x_spec = self.spectrogram_extractor(x0)
        x_spec = self.logmel_extractor(x_spec)
        frames_num = x_spec.shape[2]

        x = self.frontend(input, mixup_lambda=mixup_lambda)
        x = self._conv_backbone(x)
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)

        if self.mode in {"max", "avg"}:
            x = x.transpose(1, 2)
            x = F.relu_(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            segmentwise_output = torch.sigmoid(self.fc_audioset(x))

            if self.mode == "max":
                clipwise_output, _ = torch.max(segmentwise_output, dim=1)
            else:
                clipwise_output = torch.mean(segmentwise_output, dim=1)

            framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
            framewise_output = pad_framewise_output(framewise_output, frames_num)
            return {
                "framewise_output": framewise_output,
                "clipwise_output": clipwise_output,
            }

        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        clipwise_output, _, segmentwise_output = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        return {
            "framewise_output": framewise_output,
            "clipwise_output": clipwise_output,
        }


class Cnn14_DecisionLevelMax(_Cnn14DecisionLevelBase):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            mode="max",
        )


class Cnn14_DecisionLevelAvg(_Cnn14DecisionLevelBase):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            mode="avg",
        )


class Cnn14_DecisionLevelAtt(_Cnn14DecisionLevelBase):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            mode="att",
        )