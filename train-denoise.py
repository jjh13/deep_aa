import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl
from deep_aa.modules.bpunet import BPUNet
import torch.nn.functional as F
import torch
import random


class DenoiseModule(pl.LightningModule):
    def __init__(self, normalization, wavelet_family, noise_strength, learning_rate):
        super().__init__()
        self.model = BPUNet(wavelet_family=wavelet_family, normalization=normalization)
        self.noise_strength = noise_strength
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        x = batch

        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)

        x = (x - 0.5)
        with torch.no_grad():
            noise_stength = random.random() * self.noise_strength
            noise = torch.randn_like(x) * noise_stength

        x_hat = self(x + torch.randn_like(x) + noise)
        loss = F.l1_loss(x, x_hat)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
            for k, v in self.named_parameters():
                self.logger.experiment.add_histogram(
                    tag=k, values=v.grad, global_step=self.trainer.global_step
                )

cli = LightningCLI(model_class=DenoiseModule)


