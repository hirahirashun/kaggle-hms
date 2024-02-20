import gc
from pathlib import Path

import albumentations as A
import click
import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
#from omegaconf import OmegaConf
import yaml
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import seed_everything
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.conf import InferenceConfig, ModelConfig, TrainConfig
from src.dataset.dataset import HMSHBACDataset
from src.models.base_model import HMSSpecBaseModel
from src.models.pararell_model import HMSSpecPararellModel
from src.utils.common import process_raw_eeg_data, to_device, trace

CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
SPEC_IMG_SIZE = 512

def get_model(cfg: TrainConfig):
    #hydraが使えないせいでこんなことに
    model_name = cfg.model['name']
    backbone_name = cfg.model['params']['backbone_name']

    if model_name == "HMSSpecBaseModel":
        if cfg.data_process_ver == 2:
            in_channels = 3
        elif cfg.data_process_ver == 1:
            in_channels = 1
            if cfg.use_eeg_spec:
                in_channels += 4
        model = HMSSpecBaseModel(cfg=cfg, num_classes=cfg.num_classes, in_channels=in_channels, backbone_name=backbone_name, pretrained=False)

    elif model_name == "HMSSpecPararellModel":
        if cfg.data_process_ver == 1:
            in_channels_original_spec = 1
            in_channels_eeg_spec = 4
        elif cfg.data_process_ver == 2:
            in_channels_original_spec = 3
            in_channels_eeg_spec = 3
        elif cfg.data_process_ver == 3:
            in_channels_original_spec = 1
            in_channels_eeg_spec = 1
        
        model = HMSSpecPararellModel(cfg=cfg, 
                                      num_classes=cfg.num_classes, 
                                      in_channels_original_spec=in_channels_original_spec, 
                                      in_channels_eeg_spec=in_channels_eeg_spec,
                                      backbone_name=backbone_name,
                                      pretrained=False)
        model.is_training = False
    else:
        raise ValueError(f"Invalid model name: {cfg.model.name}")
    
    return model

def load_model(
        cfg: TrainConfig,
        fold_id: int,
        model_dir: str,
        exp_name: str,
        ):
    
    
    model = get_model(cfg=cfg)

    # load weights
    weight_path = f"{model_dir}/{exp_name}/fold_{fold_id}/best_model.pth"
    model.load_state_dict(torch.load(weight_path))
    print('load weight from "{}"'.format(weight_path))


    return model


def get_test_dataloader(
        test_df: pd.DataFrame, 
        processed_dir: str, 
        batch_size: int, 
        num_workers: int, 
        train_cfg: TrainConfig
        ) -> DataLoader:
    """get test dataloader

    Args:
        test_df: test dataframe
        cfg (DictConfig): config
        train_cfg (DictConfig): train config

    Returns:
        DataLoader: test dataloader
    """

    test_idx = test_df.index

    """Get file path and target info."""
    spec_img_pths = []
    eeg_spec_img_pths = []
    raw_eeg_pths = []

    spec_img_dir = processed_dir + "/test_spectrograms_split"
    if train_cfg.eeg_spec_version == 1:
        eeg_spec_dir = processed_dir + "/test_eeg_spectrograms_split"
    elif train_cfg.eeg_spec_version == 2:
        eeg_spec_dir = processed_dir + "/test_eeg_spectrograms_v2_split"
    raw_eeg_dir = processed_dir + "/test_raw_eeg_split"

    for eeg_id in test_df["eeg_id"].values:
        spec_img_path = spec_img_dir + f"/{eeg_id}.npy"
        spec_img_pths.append(spec_img_path)
        eeg_spec_img_path = eeg_spec_dir + f"/{eeg_id}.npy"
        eeg_spec_img_pths.append(eeg_spec_img_path)
        raw_eeg_path = raw_eeg_dir+ f"/{eeg_id}.npy"
        raw_eeg_pths.append(raw_eeg_path)
    
    labels = [0]*len(test_idx)
    level_labels = [0]*len(test_idx)

    test_data = {
        "spec_paths": [spec_img_pths[idx] for idx in test_idx],
        "eeg_spec_paths": [eeg_spec_img_pths[idx] for idx in test_idx],
        "raw_eeg_paths": [raw_eeg_pths[idx] for idx in test_idx],
        "labels": labels,
        "level_labels": level_labels
    }
    
    height = train_cfg.spec_img_size
    if train_cfg.data_process_ver == 1:
        width = train_cfg.spec_img_size 
    elif train_cfg.data_process_ver == 2:
        width = train_cfg.spec_img_size //2
    elif train_cfg.data_process_ver == 3:
        width = train_cfg.spec_img_size 

    test_spec_transform = A.Compose([
        A.Resize(p=1.0, height=height, width=width),
        ToTensorV2(p=1.0)
    ])

    test_eeg_transform = process_raw_eeg_data

    test_dataset = HMSHBACDataset(**test_data, 
                                  spec_transform=test_spec_transform, 
                                  raw_eeg_transform=test_eeg_transform, 
                                  use_eeg_spec=True, 
                                  use_raw_eeg=True, 
                                  data_process_ver=train_cfg.data_process_ver, 
                                  is_train=False)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


def inference(
    loader: DataLoader, n_folds: int, model_dir: str, exp_name: str, cfg: TrainConfig, device: torch.device, exclude_list: list
) -> np.ndarray:
    
    preds = []

    
    #cfg.model.params.pretrained = False #推論時ネット接続ができず，trueにするとerrorになるため

    for fold_id in range(n_folds):
        print(f"Fold {fold_id}")
        
        if fold_id in exclude_list:
            print(f"Fold {fold_id} is not used!")
        else:
            model = load_model(cfg, fold_id, model_dir, exp_name)
            model = model.to(device)
            model.eval()

            this_preds = []
            for batch in tqdm(loader, desc="inference"):
                with torch.no_grad():
                    x = to_device(batch, device)
                    pred = model(x).detach().cpu().numpy()
                    this_preds.append(pred)

            this_preds = np.concatenate(this_preds)

            preds.append(this_preds[None])

            del model
            torch.cuda.empty_cache()
            gc.collect()        

    preds = np.concatenate(preds)
    preds = preds.mean(axis=0)

    preds = softmax(preds, axis=1)

    return preds


def make_submission(
    preds: np.ndarray, test_df: pd.DataFrame
) -> pd.DataFrame:
    
    test_pred_df = pd.DataFrame(
    preds, columns=CLASSES
    )

    test_pred_df = pd.concat([test_df[["eeg_id"]], test_pred_df], axis=1)

    return test_pred_df

@click.command()
@click.option("--exp_name", "-e", default="dummy")
@click.option("--seed", "-s", default=1086)
@click.option("--batch_size", "-b", default=32)
@click.option("--num_workers", "-n", default=2)
@click.option('--exclude_list', '-e', default=[], multiple=True)
def main(exp_name, seed, batch_size, num_workers, exclude_list):
    seed_everything(seed)

    data_dir = "/kaggle/input/hms-harmful-brain-activity-classification"
    model_dir = "/kaggle/input/hms-model"
    processed_dir = "/kaggle/working"
    output_dir = "/kaggle/working"

    #train_cfg = OmegaConf.load(f"{model_dir}/{exp_name}/.hydra/config.yaml")

    #推論時pipが使えずhydraをインストールできないので，pyyamlで代用
    with open(f"{model_dir}/{exp_name}/.hydra/config.yaml", 'r') as file:
        train_cfg = yaml.safe_load(file)
    
    cfg = TrainConfig(**train_cfg)
 
    with trace("load test dataloader"):
        test_df = pd.read_csv(data_dir + "/test.csv")
        test_dataloader = get_test_dataloader(test_df, processed_dir, batch_size, num_workers, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #推論に使わないfoldを指定
    exclude_list = list(exclude_list)
    exclude_list = [int(x) for x in exclude_list]

    with trace("inference"):
        preds = inference(test_dataloader, train_cfg['n_folds'], model_dir, exp_name, cfg, device, exclude_list)

    with trace("make submission"):
        smpl_sub = pd.read_csv(data_dir + "/sample_submission.csv")
        sub_df = make_submission(preds=preds, test_df=test_df)

    sub_df.to_csv(f"{output_dir}/submission.csv", index=False)


if __name__ == "__main__":
    main()
