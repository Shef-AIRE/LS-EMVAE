import os
import pickle as pkl
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import util.transforms as T
from util.transforms import get_transforms_from_config, get_rand_augment_from_config
from util.misc import get_rank, get_world_size

class ECGDataset(Dataset):
    """
    Dataset class for ECG data stored as .pkl files.
    """
    _LEAD_SLICE = {
        "12lead": slice(0, 12),
        "limb_lead": slice(0, 6),
        "lead1": slice(0, 1),
        "lead2": slice(1, 2),
    }

    def __init__(
        self,
        root_dir: str,
        filenames: Iterable = None,
        labels: Optional[Iterable] = None,
        fs_list: Optional[Iterable] = None,
        target_lead: str = "12lead",
        target_fs: int = 250,
        transform: Optional[object] = None,
        label_transform: Optional[object] = None,
    ):
        """
        Args:
            root_dir (str): Directory where the .pkl data files are stored.
            filenames (list): List of filenames of the ECG data.
            labels (list, optional): Labels corresponding to each file.
            fs_list (list, optional): Sampling rates for each ECG file.
            target_lead (str): The lead(s) to extract from the ECG data.
            target_fs (int): Target sampling rate.
            transform (callable, optional): Transform to apply to the ECG data.
            label_transform (callable, optional): Transform to apply to the labels.
        """
        self.root_dir = root_dir
        self.filenames = filenames
        self.labels = labels
        self.fs_list = fs_list
        self.target_lead = target_lead
        self.resample = T.Resample(target_fs=target_fs) if fs_list is not None else None

        self.transform = transform
        self.label_transform = label_transform

        self.check_dataset()

    def check_dataset(self):
        """
        Ensure dataset integrity by checking the filenames, labels, and sampling rates.
        """
        fname_not_pkl = [f for f in self.filenames if not f.endswith('.pkl')]
        assert len(fname_not_pkl) == 0, f"Some files do not have .pkl extension. (e.g. {fname_not_pkl[0]}...)"

        fpaths = [os.path.join(self.root_dir, fname) for fname in self.filenames]
        assert all(os.path.exists(fpath) for fpath in fpaths), f"Some files do not exist. (e.g. {fpaths[0]}...)"

        if self.labels is not None:
            assert len(self.filenames) == len(self.labels), "The number of filenames and labels are different."

        if self.fs_list is not None:
            assert len(self.filenames) == len(self.fs_list), "The number of filenames and fs_list are different."

        assert self.target_lead in self._LEAD_SLICE, f"target_lead should be one of {list(self._LEAD_SLICE.keys())}"

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        """
        Load and return a sample from the dataset.
        """
        fname = self.filenames[idx]
        fpath = os.path.join(self.root_dir, fname)

        with open(fpath, 'rb') as f:
            x = pkl.load(f)

        assert isinstance(x, np.ndarray), f"Data should be numpy array. ({fpath})"
        x = x[self._LEAD_SLICE[self.target_lead]]

        if self.resample is not None:
            x = self.resample(x, self.fs_list[idx])

        if self.transform:
            print(f"Data shape before transformation: {x.shape}")
            x = self.transform(x)
            if x is None:  # Skip invalid crops
                return self.__getitem__((idx + 1) % len(self))

        if self.labels is not None:
            y = self.labels[idx]
            if self.label_transform:
                y = self.label_transform(y)
            return x, y
        else:
            return x


def build_dataset(cfg: dict, split: str) -> ECGDataset:
    """
    Build and return a dataset object for training, validation, or testing.
    """
    fname_col = cfg.get("filename_col", "FILE_NAME")
    fs_col = cfg.get("fs_col", None)
    label_col = cfg.get("label_col", None)
    target_lead = cfg.get("lead", "12lead")
    target_fs = cfg.get("fs", 250)

    index_dir = os.path.realpath(cfg["index_dir"])
    ecg_dir = os.path.realpath(cfg["ecg_dir"])

    df_name = cfg.get(f"{split}_csv", None)
    assert df_name is not None, f"{split}_csv is not defined in the config."
    df = pd.read_csv(os.path.join(index_dir, df_name))

    filenames = df[fname_col].tolist()
    fs_list = df[fs_col].astype(int).tolist() if fs_col is not None else None
    labels = df[label_col].astype(int).values if label_col is not None else None

    if split == "train":
        transforms = get_transforms_from_config(cfg["train_transforms"])
        randaug_config = cfg.get("rand_augment", {})
        if randaug_config.get("use", False):
            randaug_kwargs = randaug_config.get("kwargs", {})
            transforms.append(get_rand_augment_from_config(randaug_kwargs))
    else:
        transforms = get_transforms_from_config(cfg["eval_transforms"])

    transforms = T.Compose(transforms + [T.ToTensor()])
    label_transform = T.ToTensor(dtype=cfg["label_dtype"]) if labels is not None else None

    dataset = ECGDataset(
        ecg_dir,
        filenames=filenames,
        labels=labels,
        fs_list=fs_list,
        target_lead=target_lead,
        target_fs=target_fs,
        transform=transforms,
        label_transform=label_transform,
    )

    return dataset


def collate_fn(batch):
    """
    Custom collate function to handle empty entries in batches.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Skip empty batches
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataloader(
    dataset: Dataset,
    is_distributed: bool = False,
    dist_eval: bool = False,
    mode: Literal["train", "eval"] = "train",
    **kwargs
) -> DataLoader:
    """
    Return a DataLoader with optional distributed sampling.
    """
    is_train = mode == "train"

    if is_distributed and (is_train or dist_eval):
        num_tasks = get_world_size()
        global_rank = get_rank()
        if not is_train and len(dataset) % num_tasks != 0:
            print(
                'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added.'
            )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
    else:
        sampler = torch.utils.data.RandomSampler(dataset) if is_train else torch.utils.data.SequentialSampler(dataset)

    return DataLoader(
        dataset,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collate_fn,
        **kwargs,
    )
