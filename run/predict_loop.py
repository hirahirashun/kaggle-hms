import logging
import os
from pathlib import Path

import click
import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichModelSummary,
                                         RichProgressBar)
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.conf import TrainConfig
from src.datamodule import HMSDataModule
from src.modelmodule import HMSModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)

@click.command()
@click.option("--exp_name", "-e", default="dummy")
def main(exp_name):
    cfg : TrainConfig
    cfg = OmegaConf.load(f"/home/hiramatsu/kaggle/kaggle-hms/result/{exp_name}/.hydra/config.yaml")
    seed_everything(cfg.seed)

    save_dir = cfg.dir.save_dir
    #if os.path.exists(save_dir + '/best_scores.csv'):
    ##    print(f"{cfg.exp_name} is a completed experiment.")
    #   exit()

    os.makedirs(save_dir, exist_ok=True)

    best_scores = []
    folds = []

    for fold_id in range(cfg.n_folds):
        os.makedirs(save_dir + f"/fold_{fold_id}", exist_ok=True)
        # init lightning model
        datamodule = HMSDataModule(cfg, fold_id)
        LOGGER.info("Set Up DataModule")
        model = HMSModel(
            cfg, datamodule.val_df, fold_id
        )

        weight_path = f"{cfg.dir.output_dir}/{exp_name}/fold_{fold_id}/best_model.pth"
        model.model.load_state_dict(torch.load(weight_path))

        trainer = Trainer( 
            # env
            default_root_dir=Path.cwd(),
            # num_nodes=cfg.training.num_gpus,
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.device,
            precision=16 if cfg.trainer.use_amp else 32,
            # training
            fast_dev_run=cfg.trainer.debug,  # run only 1 train batch and 1 val batch
            max_epochs=cfg.trainer.epochs,
            max_steps=cfg.trainer.epochs * len(datamodule.train_dataloader()),
            gradient_clip_val=cfg.trainer.gradient_clip_val,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
            num_sanity_val_steps=0,
            log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
            sync_batchnorm=True,
            
        )

        trainer.predict(model=model, dataloaders=datamodule.val_dataloader())

        best_scores.append(model.best_score)
        folds.append(f"fold_{fold_id}")
        print(f'fold_{fold_id}: best_score is {model.best_score}')

        wandb.finish()

    best_scores.append(np.mean(best_scores))
    folds.append(f"mean")

    print(f'CV score is {np.mean(best_scores)}.')
    
    best_score_df = pd.DataFrame(data = {"fold_id": folds, "scores": best_scores})

    best_score_df.to_csv(save_dir + '/best_scores.csv', index=False)

    return


if __name__ == "__main__":
    main()