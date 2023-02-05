import click
import lightning as pl

import gchapero.models
import gchapero.datasets

@click.command()
def train():
    model = gchapero.models.CIFAR10Classifier(learn_rate=1e-3)
    datamodule = gchapero.datasets.CIFAR10DataModule("data", batch_size=64)

    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=5)
    trainer.fit(model, datamodule)
