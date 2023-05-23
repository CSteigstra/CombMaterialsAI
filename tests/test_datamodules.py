from pathlib import Path

import pytest
import torch

from src.data.metamaterial_datamodule import MetaMaterialDataModule

@pytest.mark.parametrize("batch_size", [32, 128])
def test_metamaterial_datamodule(batch_size):
    # TODO: Add tests for the MetamaterialDataModule
    data_dir = "data/"

    dm = MetaMaterialDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MetaMaterial").exists()
    assert Path(data_dir, "MetaMaterial", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    # num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    # assert num_datapoints == 70_000
    assert len(dm.data_train) and len(dm.data_val) and len(dm.data_test)

    batch = next(iter(dm.train_dataloader()))
    x, y, mask = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert len(mask) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert mask.dtype == torch.bool
