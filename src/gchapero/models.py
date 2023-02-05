import typing as tp
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchmetrics


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, learn_rate: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.loss = nn.CrossEntropyLoss()
        self.learn_rate = learn_rate
        self.train_acc = torchmetrics.Accuracy("multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(
        self, batch: tp.Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:
        input, target = batch
        logits = self(input)
        loss = self.loss(logits, target)
        self.train_acc(logits, target)
        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc)
        return loss

    def test_step(
        self, batch: tp.Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learn_rate)
