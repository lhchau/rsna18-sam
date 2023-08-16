from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from torchvision.datasets import DatasetFolder

# from src.data.components.cutout import Cutout


class RSNADataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        # train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        train_val_test_split: Tuple[int, int] = (24015, 2669),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `RSNA-DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.49, 0.248),
            transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop((224, 224), scale=(0.35, 1), antialias=True),
            # Cutout()
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.49, 0.248)
        ])

        self.data_dir = data_dir
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # self.data_train = DatasetFolder(f"{self.data_dir}/processed/train/", loader=self.load_file, extensions="npy", transform=self.train_transforms)
            # self.data_val = DatasetFolder(f"{self.data_dir}/processed/val/", loader=self.load_file, extensions="npy", transform=self.val_transforms)
            # self.data_test = DatasetFolder(f"{self.data_dir}/processed/test/", loader=self.load_file, extensions="npy", transform=self.val_transforms)

            trainset = DatasetFolder(f"{self.data_dir}/processed/train/", loader=self.load_file, extensions="npy", transform=self.train_transforms)
            valset = DatasetFolder(f"{self.data_dir}/processed/val/", loader=self.load_file, extensions="npy", transform=self.val_transforms)
            dataset = ConcatDataset(datasets=[trainset, valset])
            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    # def test_dataloader(self) -> DataLoader[Any]:
    #     """Create and return the test dataloader.

    #     :return: The test dataloader.
    #     """
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         shuffle=False,
    #     )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
    
    def load_file(self, path):
        return np.load(path).astype(np.float32)


if __name__ == "__main__":
    _ = RSNADataModule()
