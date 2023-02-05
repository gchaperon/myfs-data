import torch
import typing as tp
import pathlib
import lightning as pl
import torchvision
import torchvision.transforms as transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, datadir: tp.Union[str, pathlib.Path], batch_size: int) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=("datadir",))
        self.datadir = pathlib.Path(datadir)
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        torchvision.datasets.CIFAR10(self.datadir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.datadir, train=False, download=True)

    def setup(self, stage: tp.Optional[str] = None) -> None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*[[0.5] * 3] * 2)]
        )

        self.train_split = torchvision.datasets.CIFAR10(
            self.datadir, train=True, transform=transform
        )
        self.test_split = torchvision.datasets.CIFAR10(
            self.datadir, train=False, transform=transform
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_split, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_split, batch_size=self.batch_size, shuffle=False, num_wokers=4
        )
