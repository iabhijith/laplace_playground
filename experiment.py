import torch
import torch.nn as nn
import numpy as np
import random
import pytorch_lightning as pl
import fire
import os
import os.path as path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from models.simple_mlp import MLPTask, MLP
from data import dataloaders

from laplace import Laplace, marglik_training
from laplace.curvature import BackPackGGN


class Experiment:
    def __init__(
        self,
        dataset,
        split="random",
        on=None,
        batch_size=64,
        deterministic=True,
        seed=99,
    ):
        self.dataset = dataset
        self.deterministic = deterministic
        if deterministic:
            self.set_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        accelerator, device = self.device_info()
        self.accelerator = accelerator
        self.device = device
        train_dataloader, test_dataloader = dataloaders.get_dataloaders(
            dataset=self.dataset, split=split, batch_size=batch_size, shuffle=True
        )
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def experiment_mlp(
        self,
        input_dim=1,
        hidden_dim=50,
        output_dim=1,
        epochs=10,
        lr=1e-3,
        check_point="model.pt",
    ):
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.train_mlp(
            model=model, dataloader=self.train_dataloader, epochs=epochs, lr=lr
        )
        check_point = path.join("results/models", check_point)
        torch.save(model, check_point)
        loss = self.test_mlp(model=model, dataloader=self.test_dataloader)
        print(f"Test loss: {loss}")
        return model, loss

    def test_mlp(self, model, dataloader):
        model.eval()
        torch.set_grad_enabled(False)
        criterion = torch.nn.MSELoss()
        losses = []
        for X, y in dataloader:
            loss = criterion(model(X), y)
            losses.append(loss.item())
        loss = sum(losses) / len(losses)
        return loss

    def train_mlp(self, model, dataloader, epochs=10, lr=1e-3):
        model.train()
        torch.set_grad_enabled(True)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epoch_losses = []
        for i in range(epochs):
            losses = []
            for X, y in dataloader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            epoch_loss = sum(losses) / len(losses)
            if i % 10 == 0:
                print(f"Epoch {i} loss: {epoch_loss}")
            epoch_losses.append(epoch_loss)
        return losses

    def train_la(
        self,
        model,
        dataloader,
        subset_of_weights="all",
        hessian_structure="full",
        sigma_noise=1.0,
        prior_precision=1.0,
        prior_mean=0.0,
        ho=False,
        hepochs=100,
    ):
        model.train()
        torch.set_grad_enabled(True)
        la = Laplace(
            model=model,
            likelihood="regression",
            subset_of_weights=subset_of_weights,
            hessian_structure=hessian_structure,
            sigma_noise=sigma_noise,
            prior_precision=prior_precision,
            prior_mean=prior_mean,
        )
        la.fit(dataloader)
        # log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
        #     1, requires_grad=True
        # )
        # hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        # for i in range(hepochs):
        #     hyper_optimizer.zero_grad()
        #     neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        #     neg_marglik.backward()
        #     hyper_optimizer.step()
        return la

    def train_la_marglik(
        self,
        dataloader,
        hessian_structure="full",
        lr=1e-2,
        epochs=50,
        input_dim=1,
        hidden_dim=50,
        output_dim=1,
    ):
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        la, model, margliks, losses = marglik_training(
            model=model,
            train_loader=dataloader,
            likelihood="regression",
            hessian_structure=hessian_structure,
            backend=BackPackGGN,
            n_epochs=epochs,
            optimizer_kwargs={"lr": lr},
            prior_structure="scalar",
        )
        return la

    def load_model(self, check_point):
        check_point = path.join("results/models", check_point)
        return torch.load(check_point)

    def train_mlp_pl(
        self,
        batch_size=64,
        input_dim=1,
        hidden_dim=50,
        output_dim=1,
        epochs=10,
        lr=1e-3,
    ):
        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ) = dataloaders.get_dataloaders(
            dataset=self.dataset, batch_size=batch_size, shuffle=True
        )

        print("Train loader", len(train_dataloader))
        logger = TensorBoardLogger(save_dir="logs", default_hp_metric=False)
        mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        task = MLPTask(model=mlp, lr=lr)
        trainer = Trainer(
            max_epochs=epochs,
            accelerator=self.accelerator,
            deterministic=self.deterministic,
            logger=logger,
        )
        trainer.fit(
            task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
        )

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def device_info(self):
        if not torch.cuda.is_available():
            return "cpu", torch.device("cpu")
        else:
            return "gpu", torch.device("cuda:0")


if __name__ == "__main__":
    fire.Fire(Experiment)
