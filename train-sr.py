
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
from deep_aa.modules.shuffle import WaveShuffle2d
from multiprocessing import freeze_support
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl
import torchvision


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

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
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
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2, tail_pass='wave_res', wavelet_family='db2'):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.tail_type = tail_pass
        self.num_upsample = num_upsample
        self.channels_in = channels

        if tail_pass == 'wave_res':
            self.pre_out = nn.Sequential(
                nn.Conv2d(filters, (4**num_upsample - 1) * 3, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )
            self.wsh = WaveShuffle2d(4 * channels, family=wavelet_family)

        elif tail_pass == 'wave':
            self.pre_out = nn.Sequential(
                nn.Conv2d(filters, (4**num_upsample) * 3, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )
            self.wsh = WaveShuffle2d(4 * channels, family=wavelet_family)

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
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        if self.tail_type == 'wave_res' or self.tail_type == 'wave':
            po = self.pre_out(out)
            details = torch.cat([x, po], dim=-3) if self.tail_type == 'wave_res' else po
            while details.shape[-3] != self.channels_in:
                dc = details.split(self.channels_in * 4, dim=-3)
                wdc = [self.wsh(_) for _ in dc]
                details = torch.cat(wdc, dim=-3)
            return details
        else:
            out = self.upsampling(out)
            out = self.conv3(out)
            return out

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)



class SrModule(pl.LightningModule):
    def __init__(self,
                 tail_pass,
                 wavelet_family,
                 data_shape,
                 lambda_adv: float = 5e-3,
                 lambda_pixel: float = 1e-2,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 8,
                 warmup_batches: int = 500,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = GeneratorRRDB(channels=3, wavelet_family=wavelet_family, tail_pass=tail_pass)
        self.discriminator = Discriminator(input_shape=data_shape)
        self.feature_extractor = FeatureExtractor()
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss()

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs_hr, imgs_lr = batch

        # train generator
        if optimizer_idx == 0:
            self.gen_hr = self(imgs_lr)

            pred_real = self.discriminator(imgs_hr).detach()
            pred_fake = self.discriminator(self.gen_hr)

            loss_pixel = F.l1_loss(self.gen_hr, imgs_hr)
            if self.global_step < self.hparams.warmup_batches * 2:
                self.log("loss_G", loss_pixel, prog_bar=True)
                return loss_pixel

            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs_lr.shape[0], 1)
            valid = valid.type_as(imgs_lr)

            loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            gen_features = self.feature_extractor(self.gen_hr)
            real_features = self.feature_extractor(imgs_hr).detach()
            loss_content = F.l1_loss(gen_features, real_features)

            loss_G = loss_content + self.hparams.lambda_adv * loss_GAN + self.hparams.lambda_pixel * loss_pixel

            # adversarial loss is binary cross-entropy
            # g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            self.log("loss_G", loss_G, prog_bar=True)
            return loss_G

        # train discriminator
        if optimizer_idx == 1 and self.global_step >= self.hparams.warmup_batches * 2:
            # Measure discriminator's ability to classify real from generated samples

            valid = torch.ones((imgs_lr.size(0), *self.discriminator.output_shape), device=batch.device, requires_grad=False) #), requires_grad=False)
            fake = torch.zeros((imgs_lr.size(0), *self.discriminator.output_shape), device=batch.device, requires_grad=False)

            pred_real = self.discriminator(imgs_hr)
            pred_fake = self.discriminator(self.gen_hr.detach())

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

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []


from dataset.div2k import Div2kDataset

# d2k = Div2kDataset('/Volumes/john/datasets/DIV2K/DIV2K_train_HR', mode='train', cache_path='/Volumes/john/datasets/DIV2K/DIV2K_cache')
# for image in d2k:
#     hr, sr = image
#     print(hr.shape, sr.shape)
#
if __name__ == '__main__':
    freeze_support()
    cli = LightningCLI(model_class=SrModule)
#
#
# model = GeneratorRRDB(3, tail_pass='wave_res')
# print(model(torch.rand(1,3,155, 155)).shape)