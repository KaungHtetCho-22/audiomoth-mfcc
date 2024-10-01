import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torch.distributions import Beta
import timm


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)

class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)



class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = \
            Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V


class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None, teacher_preds=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        return X, Y


class AttModel(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        num_class=397,
        train_period=15.0,
        infer_period=5.0,
        in_chans=1,
        cfg=None,
        training=True,
        device=torch.device("cpu")
    ):
        super().__init__()

        self.cfg = cfg

        self.logmelspec_extractor = nn.Sequential(
            MelSpectrogram(
                sample_rate=self.cfg.sample_rate,
                n_mels=self.cfg.n_mels,
                f_min=self.cfg.fmin,
                f_max=self.cfg.fmax,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                normalized=True,
            ),
            AmplitudeToDB(top_db=80.0),
            NormalizeMelSpec(),
        )

        base_model = timm.create_model(
            backbone,
            features_only=False,
            pretrained=self.cfg.use_imagenet_weights and training,
            in_chans=self.cfg.in_channels,
        )

        layers = list(base_model.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        if "efficientnet" in self.cfg.backbone:
            dense_input = base_model.num_features
        elif "swin" in self.cfg.backbone:
            dense_input = base_model.num_features
        elif hasattr(base_model, "fc"):
            dense_input = base_model.fc.in_features
        else:
            dense_input = base_model.feature_info[-1]["num_chs"]

        self.train_period = train_period
        self.infer_period = infer_period

        self.factor = int(self.train_period / self.infer_period)
        self.mixup = Mixup(mix_beta=1)
        self.global_pool = GeM()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])
        self.head = nn.Linear(dense_input, num_class)
        
        self.training = training
        self.device = device

    def forward(self, input):

        if self.training:
            x = input['wave']
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)
            y = input["loss_target"]
        else:
            x = input['wave']
            y = input["loss_target"]
        x = x.to(self.device)
        x = self.logmelspec_extractor(x)[:, None]

        if self.training:
            if np.random.random() <= 0.5:
                y2 = torch.repeat_interleave(y, self.factor, dim=0)

                for i in range(0, x.shape[0], self.factor):
                    x[i: i + self.factor], _,  = self.mixup(
                        x[i: i + self.factor],
                        y2[i: i + self.factor],
                    )

            b, c, f, t = x.shape
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(b // self.factor, self.factor * t, c, f)

            if np.random.random() <= self.cfg.mixup_p:
                x, y = self.mixup(x, y)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 3, 1)

        x = self.backbone(x)

        if self.training:
            b, c, f, t = x.shape
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            x = x.permute(0, 2, 3, 1)

        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        logit = sum([self.head(dropout(x)) for dropout in self.dropouts]) / 5

        return {"logit": logit, "target": y}
