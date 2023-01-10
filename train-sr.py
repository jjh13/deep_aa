from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
from deep_aa.modules.shuffle import WaveShuffle2d
from multiprocessing import freeze_support
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.cli import LightningCLI
import torchvision
from deep_aa.modules.normalization import NormalizationWrapper2d

import math
from torch.optim.optimizer import Optimizer


def apply_gamma(rgb, gamma="srgb"):
    """Linear to gamma rgb.
    Assume that rgb values are in the [0, 1] range (but values outside are tolerated).
    gamma can be "srgb", a real-valued exponent, or None.
    >>> apply_gamma(torch.tensor([0.5, 0.4, 0.1]).view([1, 3, 1, 1]), 0.5).view(-1)
    tensor([0.2500, 0.1600, 0.0100])
    """
    if gamma == "srgb":
        T = 0.0031308
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, 12.92 * rgb, (1.055 * torch.pow(torch.abs(rgb1), 1 / 2.4) - 0.055))
    elif gamma is None:
        return rgb
    else:
        return torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), 1.0 / gamma)



def remove_gamma(rgb, gamma="srgb"):
    """Gamma to linear rgb.
    Assume that rgb values are in the [0, 1] range (but values outside are tolerated).
    gamma can be "srgb", a real-valued exponent, or None.
    >>> remove_gamma(apply_gamma(torch.tensor([0.001, 0.3, 0.4])))
    tensor([0.0010,  0.3000,  0.4000])
    >>> remove_gamma(torch.tensor([0.5, 0.4, 0.1]).view([1, 3, 1, 1]), 2.0).view(-1)
    tensor([0.2500, 0.1600, 0.0100])
    """
    if gamma == "srgb":
        T = 0.04045
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, rgb / 12.92, torch.pow(torch.abs(rgb1 + 0.055) / 1.055, 2.4))
    elif gamma is None:
        return rgb
    else:
        res = torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), gamma) + \
              torch.min(rgb, rgb.new_tensor(0.0)) # very important to avoid vanishing gradients
        return res

class AdamPre(Optimizer):
    """Implements Adam algorithm with prediction step.

    This class implements lookahead version of Adam Optimizer.
    The structure of class is similar to Adam class in Pytorch.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, name='NotGiven'):
        self.name = name
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamPre, self).__init__(params, defaults)
        self.stepped = False

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                    state['oldWeights'] = p.data.clone()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** min(state['step'],1022)
                bias_correction2 = 1 - beta2 ** min(state['step'],1022)
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss

    def stepLookAhead(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self.stepped:
            return loss

        self.stepped = True
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                temp_grad = p.data.sub(state['oldWeights'])
                state['oldWeights'].copy_(p.data)
                p.data.add_(temp_grad)
        return loss


    def restoreStepLookAhead(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                p.data.copy_(state['oldWeights'])
        self.stepped = False
        return loss

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2, normalization=None):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [
                nn.Conv2d(in_features, filters, 3, 1, 1, bias=True),
                NormalizationWrapper2d(filters, normalization)
            ]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2, normalization=None):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters, normalization=normalization),
            DenseResidualBlock(filters, normalization=normalization),
            DenseResidualBlock(filters, normalization=normalization)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=23, num_upsample=2, tail_pass='wave_res', wavelet_family='db2', normalization=None):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters, normalization=normalization) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks

        self.conv2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            NormalizationWrapper2d(filters, normalization)
        )

        self.tail_type = tail_pass
        self.num_upsample = num_upsample
        self.channels_in = channels

        if tail_pass == 'wave_res':
            self.pre_out = nn.Sequential(
                nn.Conv2d(filters, (4**num_upsample - 1) * 3, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )
            self.wsh = WaveShuffle2d(4 * channels, family=wavelet_family)
            # self.w_scale = torch.nn.Parameter(torch.tensor([2.]))

        elif tail_pass == 'wave':
            self.pre_out = nn.Sequential(
                nn.Conv2d(filters, (4**num_upsample) * 3, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )
            self.wsh = WaveShuffle2d(4 * channels, family=wavelet_family)
        elif tail_pass == 'slim':
            upsample_layers = [
                nn.Conv2d(filters, 16*3, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU()
            ]

            for _ in range(num_upsample):
                upsample_layers += [
                    nn.PixelShuffle(upscale_factor=2)
                ]

            self.upsampling = nn.Sequential(*upsample_layers)
            self.conv3 = nn.Identity()
        elif tail_pass == 'std_wave':
            # Upsampling layers
            print(tail_pass)
            upsample_layers = []
            for _ in range(num_upsample):
                upsample_layers += [
                    nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                    WaveShuffle2d(4 * filters, family=wavelet_family),
                ]


            self.upsampling = nn.Sequential(*upsample_layers)

            # Final output block
            self.conv3 = nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
            )
        elif tail_pass == 'skip_std_wave':
            # Upsampling layers
            print(tail_pass)

            self.pre_out = nn.Sequential(
                nn.Conv2d(filters, filters * 3, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
            upsample_layers = []
            for _ in range(num_upsample):
                if _ == 0:
                    upsample_layers += [
                        WaveShuffle2d(4 * filters, family=wavelet_family),
                    ]
                else:
                    upsample_layers += [
                        nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(),
                        WaveShuffle2d(4 * filters, family=wavelet_family),
                    ]

            self.upsampling = nn.Sequential(*upsample_layers)

            # Final output block
            self.conv3 = nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
            )
        else:
            # Upsampling layers
            upsample_layers = []
            for _ in range(num_upsample):
                upsample_layers += [
                    nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                    nn.PixelShuffle(upscale_factor=2),
                ]


            self.upsampling = nn.Sequential(*upsample_layers)

            # Final output block
            self.conv3 = nn.Sequential(
                nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
            )


    def forward(self, x):
        expected_shape = [*x.shape[0:2], x.shape[2] * (2 ** self.num_upsample), x.shape[3] * (2 ** self.num_upsample)]

        if self.tail_type == 'wave':
            for _ in range(self.num_upsample):
                x = self.wsh.pre_pad(x)

        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)

        if self.tail_type == 'wave_res' or self.tail_type == 'wave':
            out = torch.add(out1, out2)
            po = self.pre_out(out)
            # print(self.num_upsample)
            details = torch.cat([x * (2**self.num_upsample), po], dim=-3) if self.tail_type == 'wave_res' else po
            while details.shape[-3] != self.channels_in:
                dc = details.split(self.channels_in * 4, dim=-3)
                #
                wdc = [self.wsh(_, no_pad = self.tail_type == 'wave') for _ in dc]
                details = torch.cat(wdc, dim=-3)

            # print(expected_shape, details.shape)
            return details[:,:,:expected_shape[2], :expected_shape[3]]
        elif self.tail_type == 'skip_std_wave':

            po = self.pre_out(out)
            out = torch.cat([out1, po], dim=-3)
            out = self.upsampling(out)
            out = self.conv3(out)
            return out
        else:
            out = torch.add(out1, out2)
            out = self.upsampling(out)
            out = self.conv3(out)
            return out

from deep_aa.modules.conv import SafeConv2d
class Discriminator(nn.Module):
    def __init__(self, input_shape, normalization='bn', use_safe=False):
        super(Discriminator, self).__init__()


        Conv2d = SafeConv2d if use_safe else nn.Conv2d
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                # layers.append(nn.BatchNorm2d(out_filters))
                layers.append(NormalizationWrapper2d(out_filters, normalization))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(NormalizationWrapper2d(out_filters, normalization))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

from loss.lpips import LPIPS

class SrModule(pl.LightningModule):
    def __init__(self,
                 tail_pass,
                 wavelet_family,
                 data_shape,
                 lambda_adv: float = 5e-3,
                 lambda_pixel: float = 1e-2,
                 lr: float = 1e-4,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 batch_size: int = 8,
                 warmup_batches: int = 500,
                 pred_G: bool = True,
                 pred_D: bool = False,
                 generator_normalization: str = None,
                 discriminator_normalization: str = None,
                 aa_discrim: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = GeneratorRRDB(channels=3, wavelet_family=wavelet_family, tail_pass=tail_pass, normalization=generator_normalization)
        self.discriminator = Discriminator(input_shape=data_shape, normalization=discriminator_normalization, use_safe=aa_discrim)
        self.feature_extractor = LPIPS().eval()
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss()
        self.first_time = True

        self.gamma = None
        self.scale = 1.

    def forward(self, z):
        return self.generator(z)

    def validation_step(self, batch, batch_idx):
        imgs_hr, imgs_lr = batch
        imgs_hr = remove_gamma(imgs_hr, gamma=self.gamma) * self.scale
        imgs_lr = remove_gamma(imgs_lr, gamma=self.gamma) * self.scale
        img_pred = (self.generator(imgs_lr)/self.scale).clamp(0, 1)
        loss = F.l1_loss(img_pred, imgs_hr)
        self.log("val_loss", loss)

        logger = self.logger.experiment
        grid = torchvision.utils.make_grid(torch.cat([
            apply_gamma(img_pred, gamma=self.gamma),
            apply_gamma(imgs_hr/self.scale, gamma=self.gamma)
        ], dim=-2))

        logger.add_image(f"generated_images_{batch_idx}", grid, self.global_step)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs_hr, imgs_lr = batch

        imgs_hr = imgs_hr * self.scale
        imgs_lr = imgs_lr * self.scale

        # train generator
        if optimizer_idx == 0:
            self.imgs_lr = imgs_lr

            self.gen_hr = self(self.imgs_lr)

            pred_real = self.discriminator(imgs_hr).detach()
            pred_fake = self.discriminator(self.gen_hr)

            loss_pixel = F.l1_loss(self.gen_hr, imgs_hr)

            loss_content = self.feature_extractor(self.gen_hr, imgs_hr.detach()).mean()
            total_loss = loss_content + self.hparams.lambda_pixel * loss_pixel

            self.log("loss_p", loss_pixel * self.hparams.lambda_pixel, prog_bar=True)
            self.log("loss_c", loss_content , prog_bar=True)
            if self.global_step < self.hparams.warmup_batches * 2:
                self.log("loss_G", total_loss, prog_bar=True)
                return total_loss #, {'content': loss_content, 'l1': loss_pixel}

            valid = torch.ones_like(pred_fake, device=imgs_lr.device, requires_grad=False) #), requires_grad=False)
            loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)


            self.log("loss_gd", self.hparams.lambda_adv * loss_GAN , prog_bar=True)
            total_loss += self.hparams.lambda_adv * loss_GAN

            # adversarial loss is binary cross-entropy
            # g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            self.log("loss_G", total_loss, prog_bar=True)
            return total_loss

        # train discriminator
        if optimizer_idx == 1 and self.global_step >= self.hparams.warmup_batches * 2:
            # Measure discriminator's ability to classify real from generated samples

            if not self.first_time and self.hparams.pred_G:
                optim = self.g_opt
                optim.stepLookAhead()
                self.gen_hr = self(self.imgs_lr)
                optim.restoreStepLookAhead()

            self.first_time = False

            pred_real = self.discriminator(imgs_hr)
            pred_fake = self.discriminator(self.gen_hr.detach().clamp(0, self.scale))

            valid = torch.ones_like(pred_fake, device=imgs_lr.device, requires_grad=False) #), requires_grad=False)
            fake = torch.zeros_like(pred_fake, device=imgs_lr.device, requires_grad=False)

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # how well can it label as real?
            loss_D = (loss_real + loss_fake) / 2

            self.log("loss_D", loss_D, prog_bar=True)
            return loss_D
        return None

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = AdamPre(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = AdamPre(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        self.g_opt = opt_g
        self.d_opt = opt_d
        return [opt_g, opt_d], []


from dataset.div2k import Div2kDataset
if __name__ == '__main__':
    freeze_support()
    cli = LightningCLI(model_class=SrModule, )
