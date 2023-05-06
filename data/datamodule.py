# !/usr/bin/python3
# @File: datamodual.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.19.12
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.datasets import RetroAGTDataSet, CacheDataset
from torch.utils.data import ConcatDataset
import os
import os.path as osp
import torch
from tqdm import tqdm
class RetroAGTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root,
            batch_size,
            use_3d_info=False,
            fast_read=True,
            split_names=None,
            num_workers=None,
            pin_memory=True,
            shuffle=True,
            predict=True,
            dataset_type='uspto_50k'):
        super().__init__()

        exist_process = os.path.exists(os.path.join(root, 'processed/test'))
        if exist_process:
            print(f'dataset in {root} has already been processed, reading from processed directory...')
            if split_names is None and not predict:
                split_names = ["train", "valid", "test"]
            elif split_names is None:
                split_names = ['test']
        else:
            print(f'processing raw file from {root}...')
            split_names = ['train', 'valid', 'test']

        self.split_names = split_names
        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0))
#             num_workers = 32
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.dataset_dict = {}

        if dataset_type == "uspto_50k":
            max_node, max_lg_na, gate_num_size = 300, 100, 10
            for data_split in split_names:
                self.dataset_dict[data_split] = RetroAGTDataSet(root=root,
                                                                data_split=data_split,
                                                                fast_read=fast_read,
                                                                use_3d_info=use_3d_info,
                                                                max_node=max_node,
                                                                max_lg_na=max_lg_na,
                                                                max_gate_num_size=max_gate_num_size,
                                                                dataset_type=dataset_type)


        elif dataset_type == "uspto_full":
            max_node, max_lg_na, max_gate_num_size = 300, 100, 20
            for data_split in split_names:
                self.dataset_dict[data_split] = RetroAGTDataSet(root=root,
                                                                data_split=data_split,
                                                                fast_read=fast_read,
                                                                use_3d_info=use_3d_info,
                                                                max_gate_num_size=max_gate_num_size,
                                                                max_node=max_node, max_lg_na=max_lg_na,
                                                                dataset_type='full')
        elif dataset_type == "uspto_mit":
            max_node, max_lg_na, max_gate_num_size, max_regents_na = 500, 300, 20, 100
            for data_split in split_names:
                self.dataset_dict[data_split] = RetroAGTDataSet(root=root,
                                                                data_split=data_split,
                                                                fast_read=fast_read,
                                                                use_3d_info=use_3d_info,
                                                                max_gate_num_size=max_gate_num_size,
                                                                max_node=max_node, max_lg_na=max_lg_na,
                                                                max_regents_na=max_regents_na,
                                                                dataset_type='mit', known_regents=False)

        elif dataset_type == "uspto_mit_wo_regents":
            max_node, max_lg_na, max_gate_num_size, max_regents_na = 500, 300, 20, 100
            for data_split in split_names:
                self.dataset_dict[data_split] = RetroAGTDataSet(root=root,
                                                                data_split=data_split,
                                                                fast_read=fast_read,
                                                                use_3d_info=use_3d_info,
                                                                max_gate_num_size=max_gate_num_size,
                                                                max_node=max_node, max_lg_na=max_lg_na,
                                                                max_regents_na=max_regents_na,
                                                                dataset_type='mit', known_regents=False)
        else:
            raise NotImplementedError


    def train_dataloader(self):
        return DataLoader(
            self.dataset_dict['train'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_dict['valid'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_dict['test'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False
        )

    def predict_dataloader(self):
        return self.test_dataloader()
