import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, X):
        out = self.layers(X)
        return out


class MLPTask(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, X):
        out = self.model(X)
        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
