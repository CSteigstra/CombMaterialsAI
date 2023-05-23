from typing import Any, Dict, Optional, Tuple, Callable, List
import os
import warnings
import torch
from urllib.error import URLError
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets.utils import _flip_byte_order, check_integrity, download_and_extract_archive, extract_archive, verify_str_arg
import torch.utils.data as data
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import transforms
import numpy as np

class MetaMaterial(VisionDataset):
    mirrors = [
        "https://zenodo.org/record/7070963/files",
    ]

    resources = [
        ("Modescaling_classification_results.zip", "79e649816a7473310f00d69b4053dae8"),
        ("Modescaling_raw_data.zip", "b498284b233f216f339b05c9c0438c8e")
    ]

    training_file = "training.pt"
    test_file = "test.pt"
    classes = [
        "0 - zero",
        "1 - one",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: str,
        sz: int,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self.sz = sz  # sz of the grid
        # self.n_scales = n_scales  # number of scales

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load grid data
        grid_file = f"data_new_rrQR_i_n_M_{self.sz}x{self.sz}_fixn4.npy"
        data = torch.from_numpy(np.load(os.path.join(self.raw_folder, grid_file)).astype(int))

        # Load classification results, class I or C.
        label_file = f"results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{self.sz}x{self.sz}_fixn4"
        targets = torch.from_numpy(np.loadtxt(os.path.join(self.raw_folder, f"{label_file}.txt"), delimiter=',').astype(int))
        # Combine with extended experiments.
        label_ext_file = f"{label_file}_classX_extend"
        targets_ext = torch.from_numpy(np.loadtxt(os.path.join(self.raw_folder, f"{label_ext_file}.txt"), delimiter=',').astype(int))
        targets[targets_ext[:, 0]] = targets_ext

        # Ignore index column 0, and reshape data to nxn grid, and grab class from targets.
        return data[:, 1:self.sz**2+1].reshape(-1, self.sz, self.sz), targets[:, 1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        grid, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        grid, target, pad_mask = self.transforms(grid, target)

        # if self.transform is not None:
        #     grid = self.transform(grid)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return grid, target, pad_mask

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def extra_repr(self) -> str:
        return f"Split: {self.sz}"

class MetaMaterialDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_split: Tuple[float, float] = (.80, .20),
        train_sz: int = 3,
        test_sz: int = 5,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_transform: Optional[transforms.Compose] = None,
        test_transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = train_transform
        self.test_transforms = test_transform


        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 1

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        MetaMaterial(self.hparams.data_dir, self.hparams.train_sz, download=True)
        MetaMaterial(self.hparams.data_dir, self.hparams.test_sz, download=True)


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MetaMaterial(self.hparams.data_dir, sz=self.hparams.train_sz, transform=self.train_transforms)
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )

            self.data_test = MetaMaterial(self.hparams.data_dir, sz=self.hparams.test_sz, transform=self.test_transforms)

    # def _collate_fn(self, batch):
    #     """Convert a list of grids to a batch of single sized grids.

    #     Args:
    #         batch: list of samples

    #     Returns:
    #         batch: tensor of samples
    #     """
    #     batch_size = self.hparams.batch_size
    #     max_size = max([grid.shape[1] for grid, _ in batch])
    #     batch = torch.zeros(batch_size, 1, max_size, max_size)
    #     for i, (grid, _) in enumerate(batch):
    #         batch[i, :, : grid.shape[1], : grid.shape[2]] = grid
    #     return batch
        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = MetaMaterialDataModule()
