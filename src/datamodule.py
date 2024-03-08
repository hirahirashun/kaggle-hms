import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.conf import TrainConfig
from src.dataset.dataset import HMSHBACDataset
from src.utils.common import process_raw_eeg_data


###################
# Load Functions
###################
def get_path_label(train_idx, val_idx, label_list, processed_dir, eeg_spec_version, train_all: pd.DataFrame):
    """Get file path and target info."""
    spec_img_pths = []
    eeg_spec_img_pths = []
    raw_eeg_pths = []

    labels = train_all[label_list].values

    for label_id in train_all["label_id"].values:
        spec_img_path = processed_dir + f"/train_spectrograms_split/{label_id}.npy"
        spec_img_pths.append(spec_img_path)
        if eeg_spec_version > 1:
            eeg_spec_img_path = processed_dir + f"/train_eeg_spectrograms_v{eeg_spec_version}_split/{label_id}.npy"
        else:
            eeg_spec_img_path = processed_dir + f"/train_eeg_spectrograms_split/{label_id}.npy"

        eeg_spec_img_pths.append(eeg_spec_img_path)
        raw_eeg_path = processed_dir + f"/train_raw_eeg_split/{label_id}.npy"
        raw_eeg_pths.append(raw_eeg_path)

    train_data = {
        "spec_paths": [spec_img_pths[idx] for idx in train_idx],
        "eeg_spec_paths": [eeg_spec_img_pths[idx] for idx in train_idx],
        "raw_eeg_paths": [raw_eeg_pths[idx] for idx in train_idx],
        "labels": [labels[idx].astype("float32") for idx in train_idx]}

    val_data = {
        "spec_paths": [spec_img_pths[idx] for idx in val_idx],
        "eeg_spec_paths": [eeg_spec_img_pths[idx] for idx in val_idx],
        "raw_eeg_paths": [raw_eeg_pths[idx] for idx in val_idx],
        "labels": [labels[idx].astype("float32") for idx in val_idx],
        }
    
    return train_data, val_data, train_idx, val_idx


def get_spec_transforms(cfg: TrainConfig):
    height = cfg.spec_img_size
    if cfg.data_process_ver == 1:
        width = cfg.spec_img_size 
    elif cfg.data_process_ver == 2:
        width = cfg.spec_img_size //2
    elif cfg.data_process_ver == 3:
        width = cfg.spec_img_size

    spec_transform = A.Compose([
        A.Resize(p=1.0, height=height, width=width),
        ToTensorV2(p=1.0)
    ])
    
    return spec_transform


class HMSDataModule(LightningDataModule):
    def __init__(self, cfg: TrainConfig, fold_id: int):
        super().__init__()
        self.cfg = cfg
        self.data_dir = cfg.dir.data_dir
        self.processed_dir = cfg.dir.processed_dir
        self.label_list = cfg.labels
        self.label_list = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]

        split = OmegaConf.load(cfg.dir.split_dir + f"/{cfg.split_by}_fold{cfg.n_folds_total}_v{cfg.fold_ver}/fold_{fold_id}.yaml")
        train_ids = split.train_id
        val_ids = split.val_id

        all_df = pd.read_csv(self.data_dir + "/train.csv")
        # convert vote to probability
        # all_df[self.label_list] /= all_df[self.label_list].sum(axis=1).values[:, None]

        # target_level = pd.read_csv(self.processed_dir + "/target_pattern.csv")
        # target_level_pred = pd.read_csv(self.processed_dir + "/target_pred.csv")

        # all_df = all_df.merge(target_level, how='left', on='eeg_id')
        # all_df = all_df.merge(target_level_pred, how='left', on='eeg_id')

        all_df = all_df.groupby(cfg.split_by).head(1).reset_index(drop=True)

        self.train_df = all_df[all_df[cfg.split_by].isin(train_ids)]
        if self.cfg.exclude_difficult_data:
            exclude_list = np.load("/home/hiramatsu/kaggle/hms-harmful-brain-activity-classification/folds/exclude_list.npy")
            self.train_df = self.train_df[~self.train_df['label_id'].isin(exclude_list)]
            
        self.val_df = all_df[all_df[cfg.split_by].isin(val_ids)]
        self.val_df[self.label_list] /= self.val_df[self.label_list].sum(axis=1).values[:, None]

        if self.cfg.hard_sample:
            labels = all_df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].values / all_df[self.label_list].sum(axis=1).values[:, None] + 1e-5
            all_df['kl'] = torch.nn.functional.kl_div(
            torch.log(torch.tensor(labels)),
            torch.tensor([1 / 6] * 6),
            reduction='none'
            ).sum(dim=1).numpy()
            labels = self.val_df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].values / self.val_df[self.label_list].sum(axis=1).values[:, None] + 1e-5
            self.val_df['kl'] = torch.nn.functional.kl_div(
            torch.log(torch.tensor(labels)),
            torch.tensor([1 / 6] * 6),
            reduction='none'
            ).sum(dim=1).numpy()
            self.train_df = all_df[all_df['kl'] < 5.5]
            self.val_df = self.val_df[self.val_df['kl'] < 5.5]

        train_idx = self.train_df.index
        val_idx = self.val_df.index

        print(f"Train data: {len(train_idx)}, valid data: {len(val_idx)}")

        self.train_path_label, self.val_path_label, _, _ = get_path_label(train_idx=train_idx, 
                                                                val_idx=val_idx, 
                                                                label_list=self.label_list,
                                                                processed_dir=self.processed_dir,
                                                                train_all=all_df,
                                                                eeg_spec_version=self.cfg.eeg_spec_version)
        self.spec_transform = get_spec_transforms(cfg)
        self.raw_eeg_transform = process_raw_eeg_data


    def train_dataloader(self):
        train_dataset = HMSHBACDataset(**self.train_path_label, 
                                       spec_transform=self.spec_transform, 
                                       raw_eeg_transform=self.raw_eeg_transform, 
                                       use_eeg_spec=self.cfg.use_eeg_spec, 
                                       use_raw_eeg=self.cfg.use_raw_eeg, 
                                       num_samples=self.cfg.num_samples, 
                                       data_process_ver=self.cfg.data_process_ver,
                                       is_train=True, 
                                       do_horizontal_flip=self.cfg.aug.do_horizontal_flip,
                                       do_label_smoothing=self.cfg.do_label_smoothing,
                                       do_xy_masking=self.cfg.aug.do_xy_masking,
                                       cut_edge_spec=self.cfg.cut_edge_spec,
                                       cut_spec_width=self.cfg.cut_spec_width)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.cfg.batch_size, 
            num_workers=self.cfg.num_workers, 
            shuffle=True, 
            drop_last=True
            )
        
        return train_loader

    def val_dataloader(self):
        val_dataset = HMSHBACDataset(**self.val_path_label, 
                                     spec_transform=self.spec_transform, 
                                     raw_eeg_transform=self.raw_eeg_transform, 
                                     use_eeg_spec=self.cfg.use_eeg_spec, 
                                     use_raw_eeg=self.cfg.use_raw_eeg, 
                                     data_process_ver=self.cfg.data_process_ver,
                                     is_train=False)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.cfg.batch_size, 
            num_workers=self.cfg.num_workers, 
            shuffle=False, 
            drop_last=False
            )
        
        return val_loader